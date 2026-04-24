"""
다중 실험 파이프라인
- YAML 설정 기반 실험 관리
- F1 스코어 기준 best model 저장
- 추론 시간 2초 이하 제약 검증
- ONNX 자동 변환 및 타이밍 검증

Usage:
  python experiment.py --config configs/exp_A.yaml
  python experiment.py --config configs/exp_A.yaml configs/exp_B.yaml  # 순차 실행
  python experiment.py --all   # configs/ 내 전체 실험 실행
"""
import argparse
import json
import os
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import BibleDataset, AiHubDataset, EMOTIONS
from losses import build_criterion
from model import EmotionClassifier

warnings.filterwarnings('ignore')

INFER_TIME_LIMIT_MS = 2000   # 추론 시간 제약 (ms)
INFER_WARMUP        = 10     # warmup 횟수
INFER_REPEAT        = 50     # 평균 측정 횟수


# ─────────────────────────────────────────────────────────────────────
# 설정 로더
# ─────────────────────────────────────────────────────────────────────
def load_config(path: str) -> dict:
    with open(path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    # 기본값
    defaults = {
        'dataset':      'bible',         # 'bible' | 'aihub'
        'data_root':    '바이블코딩',
        'emotions':     EMOTIONS,
        'train_ratio':  0.8,
        'val_ratio':    0.1,
        'val_test_ratio': 0.5,           # aihub 전용
        'max_per_class': None,           # aihub 언더샘플링
        'backbone':     'densenet121',
        'loss':        'ce',          # 'ce' | 'focal'
        'focal_gamma': 2.0,
        'epochs':      30,
        'batch_size':  32,
        'lr':          1e-4,
        'weight_decay':1e-4,
        'scheduler':   'cosine',      # 'cosine' | 'step'
        'use_edge':    False,
        'num_workers': 0,
        'seed':        42,
        'output_dir':  None,
    }
    for k, v in defaults.items():
        cfg.setdefault(k, v)
    return cfg


# ─────────────────────────────────────────────────────────────────────
# 평가 (F1 기반)
# ─────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, criterion, device, emotions):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        total_loss += criterion(logits, labels).item() * len(labels)
        all_preds.extend(logits.argmax(1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    loss  = total_loss / len(all_labels)
    f1    = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    acc   = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    f1_per = f1_score(all_labels, all_preds, average=None, zero_division=0)
    return {
        'loss':   loss,
        'acc':    acc,
        'f1':     f1,
        'f1_per': {emotions[i]: round(float(f1_per[i]), 4) for i in range(len(emotions))},
        'preds':  all_preds,
        'labels': all_labels,
    }


# ─────────────────────────────────────────────────────────────────────
# 추론 시간 측정
# ─────────────────────────────────────────────────────────────────────
def measure_infer_time_pytorch(model, device, in_channels=3) -> float:
    model.eval()
    dummy = torch.randn(1, in_channels, 224, 224).to(device)
    with torch.no_grad():
        for _ in range(INFER_WARMUP):
            model(dummy)
        t0 = time.time()
        for _ in range(INFER_REPEAT):
            model(dummy)
    return (time.time() - t0) / INFER_REPEAT * 1000  # ms


def measure_infer_time_onnx(onnx_path: str, in_channels: int = 3) -> float:
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = 1
    sess = ort.InferenceSession(onnx_path, sess_options=opts,
                                providers=['CPUExecutionProvider'])
    dummy = np.random.randn(1, in_channels, 224, 224).astype(np.float32)
    for _ in range(INFER_WARMUP):
        sess.run(None, {'input': dummy})
    t0 = time.time()
    for _ in range(INFER_REPEAT):
        sess.run(None, {'input': dummy})
    return (time.time() - t0) / INFER_REPEAT * 1000  # ms


# ─────────────────────────────────────────────────────────────────────
# ONNX 변환
# ─────────────────────────────────────────────────────────────────────
def export_onnx(model, onnx_path: str, in_channels: int = 3) -> float:
    model.eval()
    device = next(model.parameters()).device
    dummy = torch.randn(1, in_channels, 224, 224).to(device)
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
        opset_version=17,
    )
    size_mb = os.path.getsize(onnx_path) / 1024 / 1024
    return size_mb


# ─────────────────────────────────────────────────────────────────────
# 단일 실험 실행
# ─────────────────────────────────────────────────────────────────────
def run_experiment(cfg: dict) -> dict:
    # output 디렉토리
    if cfg['output_dir'] is None:
        suffix = '_edge' if cfg['use_edge'] else ''
        cfg['output_dir'] = os.path.join(
            'output', f"{cfg['backbone']}_{cfg['loss']}{suffix}"
        )
    os.makedirs(cfg['output_dir'], exist_ok=True)

    # 설정 저장
    with open(os.path.join(cfg['output_dir'], 'config.yaml'), 'w') as f:
        yaml.dump(cfg, f, allow_unicode=True)

    torch.manual_seed(cfg['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    in_channels = 4 if cfg['use_edge'] else 3

    print(f"\n{'='*60}")
    print(f"실험: {cfg['output_dir']}")
    print(f"Device: {device} | Backbone: {cfg['backbone']} | Loss: {cfg['loss']}")
    print(f"Epochs: {cfg['epochs']} | Batch: {cfg['batch_size']} | LR: {cfg['lr']}")
    print(f"{'='*60}")

    # 데이터셋
    print('데이터 로딩...')
    if cfg.get('dataset', 'bible') == 'aihub':
        ds_kwargs = dict(
            data_root=cfg['data_root'],
            emotions=cfg['emotions'],
            val_test_ratio=cfg['val_test_ratio'],
            max_per_class=cfg['max_per_class'] or None,
            use_edge=cfg['use_edge'],
            seed=cfg['seed'],
        )
        train_ds = AiHubDataset(split='train', augment=True,  **ds_kwargs)
        val_ds   = AiHubDataset(split='val',   augment=False, **ds_kwargs)
        test_ds  = AiHubDataset(split='test',  augment=False, **ds_kwargs)
    else:
        ds_kwargs = dict(
            data_root=cfg['data_root'],
            emotions=cfg['emotions'],
            train_ratio=cfg['train_ratio'],
            val_ratio=cfg['val_ratio'],
            use_edge=cfg['use_edge'],
            seed=cfg['seed'],
        )
        train_ds = BibleDataset(split='train', augment=True,  **ds_kwargs)
        val_ds   = BibleDataset(split='val',   augment=False, **ds_kwargs)
        test_ds  = BibleDataset(split='test',  augment=False, **ds_kwargs)

    emotions = cfg['emotions']
    num_classes = len(emotions)
    print(f"Train={len(train_ds)} Val={len(val_ds)} Test={len(test_ds)} | {num_classes}클래스")
    print(f"Train 클래스별: {train_ds.class_counts()}")

    train_loader = DataLoader(train_ds, cfg['batch_size'], shuffle=True,
                              num_workers=cfg['num_workers'],
                              pin_memory=(device.type == 'cuda'))
    val_loader   = DataLoader(val_ds,   cfg['batch_size'], shuffle=False,
                              num_workers=cfg['num_workers'])
    test_loader  = DataLoader(test_ds,  cfg['batch_size'], shuffle=False,
                              num_workers=cfg['num_workers'])

    # 모델
    model = EmotionClassifier(
        num_classes, backbone=cfg['backbone'],
        pretrained=True, in_channels=in_channels,
    ).to(device)

    # 손실함수
    criterion = build_criterion(cfg['loss'], num_classes, device, cfg['focal_gamma'])

    # 옵티마이저 & 스케줄러
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay']
    )
    if cfg['scheduler'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'])
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # ── 학습 루프 ──────────────────────────────────────────────────────
    best_f1, best_epoch = 0.0, 0
    history = []

    for epoch in range(1, cfg['epochs'] + 1):
        t0 = time.time()
        model.train()
        tl, tc, tt = 0.0, 0, 0
        for imgs, labels in tqdm(train_loader, leave=False, desc=f'E{epoch:03d} train'):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            tl += loss.item() * len(labels)
            tc += (logits.detach().argmax(1) == labels).sum().item()
            tt += len(labels)

        val_res = evaluate(model, val_loader, criterion, device, emotions)
        scheduler.step()
        elapsed = time.time() - t0

        row = {
            'epoch':      epoch,
            'train_loss': round(tl / tt, 4),
            'train_acc':  round(tc / tt, 4),
            'val_loss':   round(val_res['loss'], 4),
            'val_acc':    round(val_res['acc'], 4),
            'val_f1':     round(val_res['f1'], 4),
            'elapsed':    round(elapsed, 1),
        }
        history.append(row)

        print(
            f"Epoch {epoch:3d}/{cfg['epochs']} "
            f"| train loss={tl/tt:.4f} acc={tc/tt:.3f} "
            f"| val loss={val_res['loss']:.4f} acc={val_res['acc']:.3f} f1={val_res['f1']:.4f} "
            f"| {elapsed:.1f}s"
        )

        if val_res['f1'] > best_f1:
            best_f1    = val_res['f1']
            best_epoch = epoch
            torch.save({
                'epoch':        epoch,
                'backbone':     cfg['backbone'],
                'num_classes':  num_classes,
                'emotions':     emotions,
                'in_channels':  in_channels,
                'use_edge':     cfg['use_edge'],
                'state_dict':   model.state_dict(),
                'val_f1':       val_res['f1'],
                'val_f1_per':   val_res['f1_per'],
                'val_acc':      val_res['acc'],
            }, os.path.join(cfg['output_dir'], 'best_model.pth'))
            print(f"  => Best saved (val_f1={best_f1:.4f})")

    # ── 테스트 평가 ────────────────────────────────────────────────────
    print('\n[Test 평가]')
    ckpt = torch.load(os.path.join(cfg['output_dir'], 'best_model.pth'), map_location=device, weights_only=False)
    model.load_state_dict(ckpt['state_dict'])
    test_res = evaluate(model, test_loader, criterion, device, emotions)
    print(f"Test acc={test_res['acc']:.4f} | macro F1={test_res['f1']:.4f}")
    print("클래스별 F1:", test_res['f1_per'])

    cm = confusion_matrix(test_res['labels'], test_res['preds'])
    cr = classification_report(test_res['labels'], test_res['preds'],
                                target_names=emotions, digits=4)
    print('\nClassification Report:')
    print(cr)

    # ── 추론 시간 측정 ─────────────────────────────────────────────────
    print('\n[추론 시간 측정]')
    pt_ms = measure_infer_time_pytorch(model, device, in_channels)
    print(f"PyTorch CPU: {pt_ms:.2f}ms / 이미지")

    # ONNX 변환
    onnx_path = os.path.join(cfg['output_dir'], 'model.onnx')
    onnx_mb   = export_onnx(model, onnx_path, in_channels)
    onnx_ms   = measure_infer_time_onnx(onnx_path, in_channels)
    print(f"ONNX CPU:    {onnx_ms:.2f}ms / 이미지  ({onnx_mb:.1f}MB)")

    pt_ok   = pt_ms   <= INFER_TIME_LIMIT_MS
    onnx_ok = onnx_ms <= INFER_TIME_LIMIT_MS
    print(f"제약 (≤{INFER_TIME_LIMIT_MS}ms): PyTorch={'OK' if pt_ok else 'FAIL'} | ONNX={'OK' if onnx_ok else 'FAIL'}")

    if not onnx_ok:
        print(f"  ⚠️  ONNX 추론 {onnx_ms:.1f}ms > {INFER_TIME_LIMIT_MS}ms 초과!")

    # ── 결과 저장 ──────────────────────────────────────────────────────
    result = {
        'experiment':   cfg['output_dir'],
        'backbone':     cfg['backbone'],
        'loss':         cfg['loss'],
        'use_edge':     cfg['use_edge'],
        'best_epoch':   best_epoch,
        'val_f1':       round(best_f1, 4),
        'test_acc':     round(test_res['acc'], 4),
        'test_f1':      round(test_res['f1'], 4),
        'test_f1_per':  test_res['f1_per'],
        'infer_pt_ms':  round(pt_ms, 2),
        'infer_onnx_ms':round(onnx_ms, 2),
        'onnx_mb':      round(onnx_mb, 1),
        'infer_ok':     onnx_ok,
        'confusion_matrix': cm.tolist(),
        'config':       cfg,
    }

    with open(os.path.join(cfg['output_dir'], 'result.json'), 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    with open(os.path.join(cfg['output_dir'], 'history.json'), 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print(f"\n결과 저장: {cfg['output_dir']}/result.json")

    # ── XAI 시각화 ────────────────────────────────────────────────────
    # result.json / history.json 저장 후 실행 (run_xai가 두 파일을 읽음)
    try:
        from xai import run_xai
        run_xai(model, cfg['backbone'], test_ds, emotions, cfg['output_dir'])
    except Exception as e:
        print(f"XAI 생성 중 오류 (건너뜀): {e}")

    return result


# ─────────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', nargs='+', default=None, help='YAML 설정 파일 경로(들)')
    p.add_argument('--all',    action='store_true',      help='configs/ 내 전체 실험')
    args = p.parse_args()

    configs = []
    if args.all:
        configs = sorted(Path('configs').glob('*.yaml'))
    elif args.config:
        configs = args.config
    else:
        p.print_help()
        return

    results = []
    for cfg_path in configs:
        cfg = load_config(str(cfg_path))
        result = run_experiment(cfg)
        results.append(result)

    # 전체 요약
    if len(results) > 1:
        print(f"\n{'='*70}")
        print('실험 결과 요약 (val F1 기준 정렬)')
        print(f"{'='*70}")
        print(f"{'실험':<35} {'val F1':>8} {'test F1':>8} {'ONNX ms':>8} {'OK':>4}")
        print('-'*70)
        for r in sorted(results, key=lambda x: x['val_f1'], reverse=True):
            name = os.path.basename(r['experiment'])
            print(f"{name:<35} {r['val_f1']:>8.4f} {r['test_f1']:>8.4f} "
                  f"{r['infer_onnx_ms']:>8.1f} {'OK' if r['infer_ok'] else 'NG':>4}")


if __name__ == '__main__':
    main()
