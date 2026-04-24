"""
감정 분류 모델 학습 스크립트

Usage:
  python train.py --backbone densenet121 --output_dir output/densenet121
  python train.py --backbone efficientnet_b0 --use_clahe --use_edge --output_dir output/efficientnet_b0_enhanced
"""
import argparse
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import EmotionDataset, EMOTIONS, SAMPLE_EMOTIONS
from model import EmotionClassifier


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root',      default='New_sample_3')
    p.add_argument('--new_label_root', default=None,
                   help='새 7클래스 레이블 루트 (예: 라벨링데이터-.../라벨링데이터)')
    p.add_argument('--emotions',       default=None, nargs='+',
                   help='사용할 감정 목록. 미지정 시 new_label_root 있으면 7클래스, 없으면 4클래스')
    p.add_argument('--backbone',       default='densenet121',
                   choices=['efficientnet_b0', 'densenet121', 'densenet169',
                            'resnet18', 'resnet50'])
    p.add_argument('--epochs',         type=int, default=30)
    p.add_argument('--batch_size',     type=int, default=32)
    p.add_argument('--lr',             type=float, default=1e-4)
    p.add_argument('--image_size',     type=int, default=224)
    p.add_argument('--val_ratio',      type=float, default=0.2)
    p.add_argument('--output_dir',     default=None)
    p.add_argument('--num_workers',    type=int, default=0)
    p.add_argument('--use_clahe',      action='store_true')
    p.add_argument('--use_edge',       action='store_true')
    p.add_argument('--use_align',      action='store_true')
    return p.parse_args()


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in tqdm(loader, leave=False, desc='  train'):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        correct += (model(imgs).detach().argmax(1) == labels).sum().item() if False else \
                   (imgs.shape[0] - (model(imgs).detach().argmax(1) != labels).sum().item()) # reuse forward
        total += len(labels)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in tqdm(loader, leave=False, desc='  val  '):
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        total_loss += criterion(logits, labels).item() * len(labels)
        correct += (logits.argmax(1) == labels).sum().item()
        total += len(labels)
    return total_loss / total, correct / total


def main():
    args = parse_args()

    # output_dir 자동 분기
    if args.output_dir is None:
        suffix = ''
        if args.use_clahe or args.use_edge or args.use_align:
            parts = []
            if args.use_clahe:  parts.append('clahe')
            if args.use_edge:   parts.append('edge')
            if args.use_align:  parts.append('align')
            suffix = '_' + '_'.join(parts)
        args.output_dir = os.path.join('output', args.backbone + suffix)

    os.makedirs(args.output_dir, exist_ok=True)
    # 감정 목록 결정
    if args.emotions:
        use_emotions = args.emotions
    elif args.new_label_root:
        use_emotions = EMOTIONS          # 7클래스 시도 (이미지 없는 클래스는 자동 스킵)
    else:
        use_emotions = SAMPLE_EMOTIONS   # 기존 4클래스

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    in_channels = 4 if args.use_edge else 3

    print(f'Device: {device}')
    print(f'Backbone: {args.backbone}  |  in_channels: {in_channels}')
    print(f'전처리: clahe={args.use_clahe}, edge={args.use_edge}, align={args.use_align}')
    print(f'감정 클래스 (요청): {use_emotions}')
    print(f'Output: {args.output_dir}')

    ds_kwargs = dict(
        data_root=args.data_root,
        new_label_root=args.new_label_root,
        emotions=use_emotions,
        val_ratio=args.val_ratio,
        image_size=args.image_size,
        use_clahe=args.use_clahe,
        use_edge=args.use_edge,
        use_align=args.use_align,
    )
    train_ds = EmotionDataset(split='train', augment=True,  **ds_kwargs)
    val_ds   = EmotionDataset(split='val',   augment=False, **ds_kwargs)

    # 실제 로드된 클래스만 사용
    loaded_emotions = [e for e in use_emotions if e in train_ds.class_counts()]
    num_classes = len(loaded_emotions)
    if num_classes == 0:
        raise RuntimeError('로드된 이미지가 없습니다. 데이터 경로를 확인하세요.')

    print(f'\n실제 로드된 클래스 ({num_classes}개): {loaded_emotions}')
    print(f'Train: {len(train_ds)}장  |  Val: {len(val_ds)}장')
    print('클래스별:', train_ds.class_counts())

    # label 인덱스를 로드된 클래스 기준으로 재매핑
    if loaded_emotions != use_emotions:
        label_map = {use_emotions.index(e): i for i, e in enumerate(loaded_emotions)}
        for s in train_ds.samples + val_ds.samples:
            s['label'] = label_map[s['label']]

    counts = train_ds.class_counts()
    class_weights = torch.tensor(
        [1.0 / counts.get(e, 1) for e in loaded_emotions], dtype=torch.float32
    ).to(device)
    class_weights = class_weights / class_weights.sum() * num_classes

    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=(device.type=='cuda'))
    val_loader   = DataLoader(val_ds,   args.batch_size, shuffle=False,
                              num_workers=args.num_workers)

    model = EmotionClassifier(
        num_classes, backbone=args.backbone,
        pretrained=True, in_channels=in_channels,
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # 학습 루프 (forward 중복 방지)
        model.train()
        tl, tc, tt = 0.0, 0, 0
        for imgs, labels in tqdm(train_loader, leave=False, desc='  train'):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            tl += loss.item() * len(labels)
            tc += (logits.detach().argmax(1) == labels).sum().item()
            tt += len(labels)

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(
            f'Epoch {epoch:3d}/{args.epochs} '
            f'| train loss {tl/tt:.4f} acc {tc/tt:.3f} '
            f'| val loss {val_loss:.4f} acc {val_acc:.3f} '
            f'| {time.time()-t0:.1f}s'
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'backbone': args.backbone,
                'num_classes': num_classes,
                'emotions': loaded_emotions,
                'in_channels': in_channels,
                'use_clahe': args.use_clahe,
                'use_edge': args.use_edge,
                'use_align': args.use_align,
                'state_dict': model.state_dict(),
                'val_acc': val_acc,
            }, os.path.join(args.output_dir, 'best_model.pth'))
            print(f'  => Best saved (val_acc={val_acc:.3f})')

    print(f'\n완료. Best val acc: {best_val_acc:.3f} → {args.output_dir}/best_model.pth')


if __name__ == '__main__':
    main()
