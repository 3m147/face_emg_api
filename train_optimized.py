import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import f1_score
import numpy as np
import os

# 1. 스터디 공통 전처리 (이미 224x224 크롭은 완료됨)
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(), # 데이터 증강 (좌우 반전)
        transforms.RandomRotation(15), # 데이터 증강 (회전)
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # 데이터 증강 (색상 및 명암)
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# 2. 내 PC에 맞는 데이터 경로 설정 (이미 크롭된 샘플데이터 사용)
data_dir = r'd:\한국표정사진\샘플데이터'

# 전체 데이터를 불러온 뒤, 학습용(Train)과 검증용(Val)으로 8:2 분할
full_dataset = datasets.ImageFolder(data_dir)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# 분할된 데이터셋에 각각 트랜스폼 적용을 위한 래퍼 클래스 처리 (간략화)
train_dataset.dataset.transform = data_transforms['train']
val_dataset.dataset.transform = data_transforms['val']

dataloaders = {
    # 윈도우 환경 오류 방지를 위해 num_workers=0 으로 수정
    'train': DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0),
    'val': DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
}

# 3. 모델 로드 및 설정 (레이턴시 200ms 최적화를 위해 경량화된 ResNet-18 사용)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# 감정 클래스 개수(예: 당황, 기쁨, 분노, 슬픔 -> 4개)에 맞게 마지막 노드 수정
num_classes = len(full_dataset.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

print(f"인식할 감정 종류: {full_dataset.classes}")

# 손실 함수 및 최적화 기법 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 30 # 성능 향상을 위해 30으로 증가

# 윈도우 환경(멀티프로세싱) 구동 시 무한루프 방지를 위한 시작점 
if __name__ == '__main__':
    # 성능이 가장 높았던 F1 Score를 기억할 변수
    best_f1 = 0.0
    
    # 학습률 스케줄러 (성능 개선이 없을 때 학습률 감소)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            all_preds = []
            all_labels = []

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            # F1 Score 계산 (다중 클래스이므로 'macro' 또는 'weighted' 사용)
            epoch_f1 = f1_score(all_labels, all_preds, average='weighted')

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} | F1 Score: {epoch_f1:.4f}')

            # Val 단계에서 스케줄러 업데이트 및 최고 모델 갱신 저장
            if phase == 'val':
                scheduler.step(epoch_f1)
                
                if epoch_f1 > best_f1:
                    best_f1 = epoch_f1
                    # 4. 금요일/화요일 산출물을 위해 가중치 파일 저장
                    torch.save(model.state_dict(), r'd:\한국표정사진\resnet18_emotion_best.pth')
                    print(f"  --> ⭐ Best F1 Score 갱신! ({best_f1:.4f}) 모델 가중치가 저장되었습니다.")

    print(f'\n학습 완료! 최종 최고 F1 Score: {best_f1:.4f}')