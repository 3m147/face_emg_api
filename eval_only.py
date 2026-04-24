"""
학습 없이 평가+ONNX+XAI만 실행 (best_model.pth 존재 시)
Usage: python eval_only.py --config configs/exp_A_densenet121_ce.yaml
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import yaml
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from dataset import BibleDataset
from experiment import (load_config, evaluate, export_onnx,
                        measure_infer_time_pytorch, measure_infer_time_onnx,
                        INFER_TIME_LIMIT_MS)
from losses import build_criterion
from model import EmotionClassifier


def eval_only(cfg: dict):
    pth = os.path.join(cfg['output_dir'], 'best_model.pth')
    if not os.path.exists(pth):
        print(f"best_model.pth 없음: {pth}")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    in_channels = 4 if cfg['use_edge'] else 3
    emotions = cfg['emotions']
    num_classes = len(emotions)

    # 데이터
    test_ds = BibleDataset(split='test', augment=False,
                           data_root=cfg['data_root'], emotions=emotions,
                           train_ratio=cfg['train_ratio'], val_ratio=cfg['val_ratio'],
                           use_edge=cfg['use_edge'], seed=cfg['seed'])
    test_loader = DataLoader(test_ds, cfg['batch_size'], shuffle=False,
                             num_workers=cfg['num_workers'])

    # 모델 로드
    model = EmotionClassifier(num_classes, backbone=cfg['backbone'],
                               pretrained=False, in_channels=in_channels).to(device)
    ckpt = torch.load(pth, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['state_dict'])
    best_f1 = ckpt.get('val_f1', 0.0)
    best_epoch = ckpt.get('epoch', 0)

    criterion = build_criterion(cfg['loss'], num_classes, device, cfg['focal_gamma'])

    # 테스트 평가
    print(f"\n[Test 평가] {cfg['output_dir']}")
    test_res = evaluate(model, test_loader, criterion, device, emotions)
    print(f"Test acc={test_res['acc']:.4f} | macro F1={test_res['f1']:.4f}")
    print("클래스별 F1:", test_res['f1_per'])

    cm = confusion_matrix(test_res['labels'], test_res['preds'])
    cr = classification_report(test_res['labels'], test_res['preds'],
                                target_names=emotions, digits=4)
    print('\nClassification Report:')
    print(cr)

    # 추론 시간
    print('\n[추론 시간 측정]')
    pt_ms = measure_infer_time_pytorch(model, device, in_channels)
    print(f"PyTorch CPU: {pt_ms:.2f}ms")

    onnx_path = os.path.join(cfg['output_dir'], 'model.onnx')
    onnx_mb = export_onnx(model, onnx_path, in_channels)
    onnx_ms = measure_infer_time_onnx(onnx_path, in_channels)
    print(f"ONNX CPU:    {onnx_ms:.2f}ms ({onnx_mb:.1f}MB)")

    onnx_ok = onnx_ms <= INFER_TIME_LIMIT_MS
    print(f"제약 (≤{INFER_TIME_LIMIT_MS}ms): {'OK' if onnx_ok else 'FAIL'}")

    # 결과 저장
    result = {
        'experiment':    cfg['output_dir'],
        'backbone':      cfg['backbone'],
        'loss':          cfg['loss'],
        'use_edge':      cfg['use_edge'],
        'best_epoch':    best_epoch,
        'val_f1':        round(best_f1, 4),
        'test_acc':      round(test_res['acc'], 4),
        'test_f1':       round(test_res['f1'], 4),
        'test_f1_per':   test_res['f1_per'],
        'infer_pt_ms':   round(pt_ms, 2),
        'infer_onnx_ms': round(onnx_ms, 2),
        'onnx_mb':       round(onnx_mb, 1),
        'infer_ok':      onnx_ok,
        'confusion_matrix': cm.tolist(),
        'config':        cfg,
    }
    with open(os.path.join(cfg['output_dir'], 'result.json'), 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # XAI
    try:
        from xai import run_xai
        run_xai(model, cfg['backbone'], test_ds, emotions, cfg['output_dir'])
    except Exception as e:
        print(f"XAI 오류 (건너뜀): {e}")

    print(f"\n완료: {cfg['output_dir']}")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', nargs='+', required=True)
    args = p.parse_args()
    for cfg_path in args.config:
        cfg = load_config(cfg_path)
        eval_only(cfg)


if __name__ == '__main__':
    main()
