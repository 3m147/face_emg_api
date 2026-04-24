"""
XAI (Explainable AI) 모듈
- GradCAM : 클래스 활성화 맵 (어느 영역을 보고 판단했는가)
- GradCAM++: 더 정확한 공간적 위치
- 클래스별 대표 샘플 시각화
- 학습 과정 시각화 (loss, F1, confusion matrix)

Usage:
  from xai import run_xai
  run_xai(model, test_dataset, emotions, output_dir)
"""
import json
import os

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD  = [0.229, 0.224, 0.225]


# ─────────────────────────────────────────────────────────────────────
# GradCAM
# ─────────────────────────────────────────────────────────────────────
class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.
    target_layer: 마지막 conv 레이어 이름 (backbone별 자동 감지)
    """
    def __init__(self, model: torch.nn.Module, backbone: str):
        self.model       = model
        self.backbone    = backbone
        self._fmap       = None
        self._grad       = None
        self._hook_f     = None
        self._hook_b     = None
        self._register(backbone)

    def _get_target_layer(self, backbone: str):
        net = self.model.net
        if backbone == 'densenet121':
            return net.features.denseblock4
        elif backbone == 'densenet169':
            return net.features.denseblock4
        elif backbone == 'efficientnet_b0':
            return net.features[-1]
        elif backbone in ('resnet18', 'resnet50'):
            return net.layer4
        else:
            raise ValueError(f'지원하지 않는 backbone: {backbone}')

    def _register(self, backbone: str):
        layer = self._get_target_layer(backbone)
        self._hook_f = layer.register_forward_hook(
            lambda m, i, o: setattr(self, '_fmap', o.detach())
        )
        self._hook_b = layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, '_grad', go[0].detach())
        )

    def remove(self):
        self._hook_f.remove()
        self._hook_b.remove()

    def __call__(self, x: torch.Tensor, class_idx: int = None) -> np.ndarray:
        """
        x: (1, C, H, W) tensor
        반환: (H, W) numpy [0,1] 히트맵
        """
        self.model.eval()
        x = x.requires_grad_(True)
        logits = self.model(x)

        if class_idx is None:
            class_idx = logits.argmax(1).item()

        self.model.zero_grad()
        logits[0, class_idx].backward()

        # GAP over gradients
        weights = self._grad.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)
        cam = (weights * self._fmap).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        cam = cv2.resize(cam, (224, 224))
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam


class GradCAMPP(GradCAM):
    """GradCAM++ — 더 정밀한 위치 추정."""
    def __call__(self, x: torch.Tensor, class_idx: int = None) -> np.ndarray:
        self.model.eval()
        x = x.requires_grad_(True)
        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(1).item()

        self.model.zero_grad()
        score = logits[0, class_idx]
        score.backward()

        grad  = self._grad                          # (1, C, h, w)
        fmap  = self._fmap                          # (1, C, h, w)
        grad2 = grad ** 2
        grad3 = grad ** 3
        alpha_numer = grad2
        alpha_denom = 2 * grad2 + fmap * grad3.sum(dim=(2,3), keepdim=True) + 1e-7
        alpha = alpha_numer / alpha_denom
        weights = (alpha * F.relu(grad)).sum(dim=(2, 3), keepdim=True)

        cam = (weights * fmap).sum(dim=1, keepdim=True)
        cam = F.relu(cam).squeeze().cpu().numpy()
        cam = cv2.resize(cam, (224, 224))
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam


# ─────────────────────────────────────────────────────────────────────
# 이미지 → tensor
# ─────────────────────────────────────────────────────────────────────
_to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(IMG_MEAN, IMG_STD),
])

def img_to_tensor(img_rgb: np.ndarray) -> torch.Tensor:
    return _to_tensor(img_rgb).unsqueeze(0)

def denormalize(tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(IMG_MEAN).view(3,1,1)
    std  = torch.tensor(IMG_STD).view(3,1,1)
    img  = tensor.squeeze().cpu() * std + mean
    return img.permute(1,2,0).clamp(0,1).numpy()


# ─────────────────────────────────────────────────────────────────────
# 히트맵 오버레이
# ─────────────────────────────────────────────────────────────────────
def overlay_heatmap(img_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (img_rgb * (1 - alpha) + heatmap * alpha).astype(np.uint8)
    return overlay


# ─────────────────────────────────────────────────────────────────────
# 클래스별 GradCAM 시각화
# ─────────────────────────────────────────────────────────────────────
def visualize_gradcam_per_class(
    model, backbone: str, dataset, emotions: list,
    output_dir: str, n_samples: int = 5, method: str = 'gradcam'
):
    """
    각 감정 클래스별 n_samples 이미지에 대해 GradCAM 시각화.
    - 원본 / GradCAM / GradCAM++ 3열 비교
    """
    os.makedirs(output_dir, exist_ok=True)
    device = next(model.parameters()).device

    cam_fn  = GradCAMPP(model, backbone) if method == 'gradcam++' else GradCAM(model, backbone)
    cam_pp  = GradCAMPP(model, backbone)

    # 클래스별 샘플 인덱스
    class_indices = {e: [] for e in emotions}
    for i, s in enumerate(dataset.samples):
        e = s['emotion']
        if len(class_indices[e]) < n_samples:
            class_indices[e].append(i)

    for emotion in emotions:
        idxs = class_indices[emotion]
        if not idxs:
            continue

        fig, axes = plt.subplots(len(idxs), 4, figsize=(16, 4 * len(idxs)))
        if len(idxs) == 1:
            axes = axes[np.newaxis, :]

        fig.suptitle(f'GradCAM 분석 — {emotion}', fontsize=15, fontweight='bold')

        for row, idx in enumerate(idxs):
            tensor, label = dataset[idx]
            x = tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(x)
            probs     = torch.softmax(logits, dim=1)[0].cpu().numpy()
            pred_idx  = probs.argmax()
            pred_name = emotions[pred_idx]
            correct   = (pred_idx == label)

            # 원본 이미지 복원 (edge 채널 있으면 RGB만)
            img_np = denormalize(tensor[:3].unsqueeze(0))
            img_u8 = (img_np * 255).astype(np.uint8)

            # GradCAM (예측 클래스)
            cam_pred  = cam_fn(x.clone(), class_idx=pred_idx)
            # GradCAM++ (정답 클래스)
            cam_true  = cam_pp(x.clone(), class_idx=label)

            # col 0: 원본
            axes[row, 0].imshow(img_u8)
            axes[row, 0].set_title(
                f'원본\n정답: {emotion} | 예측: {pred_name} {"✓" if correct else "✗"}',
                fontsize=9, color='green' if correct else 'red'
            )
            axes[row, 0].axis('off')

            # col 1: GradCAM (예측 기준)
            axes[row, 1].imshow(overlay_heatmap(img_u8, cam_pred))
            axes[row, 1].set_title(f'GradCAM\n(예측: {pred_name})', fontsize=9)
            axes[row, 1].axis('off')

            # col 2: GradCAM++ (정답 기준)
            axes[row, 2].imshow(overlay_heatmap(img_u8, cam_true))
            axes[row, 2].set_title(f'GradCAM++\n(정답: {emotion})', fontsize=9)
            axes[row, 2].axis('off')

            # col 3: 신뢰도 바
            colors = ['#ef4444' if i == pred_idx else
                      '#22c55e' if i == label else '#94a3b8'
                      for i in range(len(emotions))]
            bars = axes[row, 3].barh(emotions, probs, color=colors)
            axes[row, 3].set_xlim(0, 1)
            axes[row, 3].set_title('예측 신뢰도', fontsize=9)
            axes[row, 3].tick_params(labelsize=8)
            for bar, p in zip(bars, probs):
                axes[row, 3].text(p + 0.01, bar.get_y() + bar.get_height()/2,
                                  f'{p:.2f}', va='center', fontsize=7)

        plt.tight_layout()
        save_path = os.path.join(output_dir, f'gradcam_{emotion}.png')
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close()
        print(f'  저장: {save_path}')

    cam_fn.remove()
    cam_pp.remove()


# ─────────────────────────────────────────────────────────────────────
# 예측 오류 케이스 시각화
# ─────────────────────────────────────────────────────────────────────
def visualize_errors(
    model, backbone: str, dataset, emotions: list,
    output_dir: str, n_per_pair: int = 3
):
    """혼동 쌍별 오분류 샘플 GradCAM 시각화."""
    os.makedirs(output_dir, exist_ok=True)
    device   = next(model.parameters()).device
    cam_fn   = GradCAM(model, backbone)

    errors = {}   # (true, pred) → list of indices
    model.eval()
    for i, s in enumerate(dataset.samples):
        tensor, label = dataset[i]
        x = tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(x).argmax(1).item()
        if pred != label:
            key = (emotions[label], emotions[pred])
            errors.setdefault(key, []).append(i)

    # 가장 빈번한 혼동 쌍 상위 6개
    top_pairs = sorted(errors.items(), key=lambda x: len(x[1]), reverse=True)[:6]
    if not top_pairs:
        print('  오분류 없음')
        cam_fn.remove()
        return

    fig, axes = plt.subplots(len(top_pairs), n_per_pair,
                             figsize=(5 * n_per_pair, 5 * len(top_pairs)))
    if len(top_pairs) == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle('오분류 케이스 GradCAM', fontsize=14, fontweight='bold')

    for row, ((true_e, pred_e), idxs) in enumerate(top_pairs):
        for col in range(n_per_pair):
            ax = axes[row, col]
            if col >= len(idxs):
                ax.axis('off')
                continue
            tensor, label = dataset[idxs[col]]
            x = tensor.unsqueeze(0).to(device)
            with torch.no_grad():
                probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()
            cam = cam_fn(x.clone(), class_idx=emotions.index(pred_e))
            img_np = denormalize(tensor[:3].unsqueeze(0))
            img_u8 = (img_np * 255).astype(np.uint8)
            ax.imshow(overlay_heatmap(img_u8, cam))
            ax.set_title(
                f'정답:{true_e} → 예측:{pred_e}\n'
                f'신뢰도: {probs[emotions.index(pred_e)]:.2f}',
                fontsize=8
            )
            ax.axis('off')
        # 행 레이블
        axes[row, 0].set_ylabel(f'{true_e}→{pred_e}\n({len(idxs)}건)',
                                 fontsize=9, rotation=0, labelpad=60)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'error_analysis.png')
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    cam_fn.remove()
    print(f'  오류 분석 저장: {save_path}')


# ─────────────────────────────────────────────────────────────────────
# 학습 과정 시각화
# ─────────────────────────────────────────────────────────────────────
def visualize_training(history_path: str, output_dir: str):
    """history.json → 학습 곡선 저장."""
    with open(history_path, encoding='utf-8') as f:
        history = json.load(f)

    epochs     = [h['epoch']      for h in history]
    train_loss = [h['train_loss'] for h in history]
    val_loss   = [h['val_loss']   for h in history]
    train_acc  = [h['train_acc']  for h in history]
    val_acc    = [h['val_acc']    for h in history]
    val_f1     = [h['val_f1']     for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('학습 과정', fontsize=14, fontweight='bold')

    # Loss
    axes[0].plot(epochs, train_loss, label='Train', color='#3b82f6', linewidth=2)
    axes[0].plot(epochs, val_loss,   label='Val',   color='#ef4444', linewidth=2)
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, train_acc, label='Train', color='#3b82f6', linewidth=2)
    axes[1].plot(epochs, val_acc,   label='Val',   color='#ef4444', linewidth=2)
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[1].set_ylim(0, 1)

    # Val F1
    best_ep = epochs[val_f1.index(max(val_f1))]
    axes[2].plot(epochs, val_f1, color='#8b5cf6', linewidth=2, label='Val Macro F1')
    axes[2].axvline(x=best_ep, color='#f59e0b', linestyle='--', linewidth=1.5,
                    label=f'Best Epoch={best_ep} F1={max(val_f1):.4f}')
    axes[2].set_title('Val Macro F1')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('F1')
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    axes[2].set_ylim(0, 1)

    plt.tight_layout()
    path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'  학습 곡선 저장: {path}')


# ─────────────────────────────────────────────────────────────────────
# Confusion Matrix 시각화
# ─────────────────────────────────────────────────────────────────────
def visualize_confusion_matrix(result_path: str, output_dir: str):
    with open(result_path, encoding='utf-8') as f:
        result = json.load(f)

    emotions = list(result['test_f1_per'].keys())
    cm = np.array(result['confusion_matrix'])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        f"혼동 행렬 — {os.path.basename(result['experiment'])}\n"
        f"Test F1={result['test_f1']:.4f}  Acc={result['test_acc']:.4f}",
        fontsize=13, fontweight='bold'
    )

    for ax, mat, title, fmt in [
        (axes[0], cm,      '절대값',  'd'),
        (axes[1], cm_norm, '비율(%)', '.2f'),
    ]:
        im = ax.imshow(mat, cmap='Blues')
        ax.set_xticks(range(len(emotions)))
        ax.set_yticks(range(len(emotions)))
        ax.set_xticklabels(emotions, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(emotions, fontsize=9)
        ax.set_xlabel('예측', fontsize=10)
        ax.set_ylabel('정답', fontsize=10)
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046)
        for i in range(len(emotions)):
            for j in range(len(emotions)):
                val = mat[i, j]
                text = f'{int(val)}' if fmt == 'd' else f'{val:.2f}'
                color = 'white' if (fmt == 'd' and val > cm.max()*0.6) or \
                                   (fmt == '.2f' and val > 0.6) else 'black'
                ax.text(j, i, text, ha='center', va='center',
                        fontsize=8, color=color)

    plt.tight_layout()
    path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'  혼동 행렬 저장: {path}')


# ─────────────────────────────────────────────────────────────────────
# 클래스별 F1 레이더 차트
# ─────────────────────────────────────────────────────────────────────
def visualize_f1_radar(result_path: str, output_dir: str):
    with open(result_path, encoding='utf-8') as f:
        result = json.load(f)

    emotions = list(result['test_f1_per'].keys())
    f1_vals  = [result['test_f1_per'][e] for e in emotions]
    N = len(emotions)
    angles = [2 * np.pi * i / N for i in range(N)] + [0]
    f1_vals_plot = f1_vals + [f1_vals[0]]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.plot(angles, f1_vals_plot, 'o-', linewidth=2, color='#4f46e5')
    ax.fill(angles, f1_vals_plot, alpha=0.25, color='#4f46e5')
    ax.set_thetagrids(np.degrees(angles[:-1]), emotions, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2','0.4','0.6','0.8','1.0'], fontsize=8)
    ax.set_title(
        f"클래스별 F1 — {os.path.basename(result['experiment'])}\n"
        f"Macro F1 = {result['test_f1']:.4f}",
        fontsize=12, fontweight='bold', pad=20
    )
    ax.grid(True, alpha=0.4)

    path = os.path.join(output_dir, 'f1_radar.png')
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'  F1 레이더 저장: {path}')


# ─────────────────────────────────────────────────────────────────────
# 전체 XAI 실행 진입점
# ─────────────────────────────────────────────────────────────────────
def run_xai(model, backbone: str, test_dataset, emotions: list, output_dir: str):
    """experiment.py에서 학습 완료 후 호출."""
    xai_dir = os.path.join(output_dir, 'xai')
    os.makedirs(xai_dir, exist_ok=True)
    print('\n[XAI 시각화 시작]')

    # 1. 학습 곡선
    history_path = os.path.join(output_dir, 'history.json')
    if os.path.exists(history_path):
        visualize_training(history_path, xai_dir)

    # 2. 혼동 행렬
    result_path = os.path.join(output_dir, 'result.json')
    if os.path.exists(result_path):
        visualize_confusion_matrix(result_path, xai_dir)
        visualize_f1_radar(result_path, xai_dir)

    # 3. GradCAM 클래스별
    print('  GradCAM 생성 중...')
    visualize_gradcam_per_class(model, backbone, test_dataset, emotions,
                                os.path.join(xai_dir, 'gradcam'), n_samples=4)

    # 4. 오분류 분석
    print('  오분류 분석 중...')
    visualize_errors(model, backbone, test_dataset, emotions,
                     os.path.join(xai_dir, 'errors'), n_per_pair=3)

    print(f'[XAI 완료] 저장 위치: {xai_dir}/')


# ─────────────────────────────────────────────────────────────────────
# 독립 실행
# ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse
    from dataset import BibleDataset
    from model import EmotionClassifier

    p = argparse.ArgumentParser()
    p.add_argument('--output_dir', required=True, help='실험 결과 디렉토리')
    args = p.parse_args()

    result_p = os.path.join(args.output_dir, 'result.json')
    with open(result_p, encoding='utf-8') as f:
        result = json.load(f)

    cfg      = result['config']
    emotions = cfg['emotions']
    backbone = cfg['backbone']
    in_ch    = 4 if cfg['use_edge'] else 3
    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = EmotionClassifier(len(emotions), backbone=backbone,
                               pretrained=False, in_channels=in_ch).to(device)
    ckpt  = torch.load(os.path.join(args.output_dir, 'best_model.pth'), map_location=device, weights_only=False)
    model.load_state_dict(ckpt['state_dict'])

    test_ds = BibleDataset(
        data_root=cfg['data_root'], emotions=emotions,
        split='test', train_ratio=cfg['train_ratio'],
        val_ratio=cfg['val_ratio'], use_edge=cfg['use_edge'],
        seed=cfg['seed'],
    )

    run_xai(model, backbone, test_ds, emotions, args.output_dir)
