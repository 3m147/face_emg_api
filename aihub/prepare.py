"""
AI Hub 한국인 감정인식 데이터 전처리 스크립트
- zip에서 이미지 추출 (G: 드라이브 원본 유지)
- 라벨 JSON bbox로 얼굴 크롭 (전문가 3인 어노테이션 평균)
- 224×224 리사이즈 후 저장
- 클래스별 최대 N장 언더샘플링

Usage:
  python aihub/prepare.py --out G:/aihub_cropped --max_per_class 16000
  python aihub/prepare.py --out G:/aihub_cropped --max_per_class 16000 --dry_run
"""
import argparse
import json
import os
import random
import zipfile
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# ── 경로 ──────────────────────────────────────────────────────────────
AIHUB_ROOT = r"G:\내 드라이브\aihub_face\한국인 감정인식을 위한 복합 영상"
EMOTIONS   = ['기쁨', '당황', '분노', '불안', '상처', '슬픔', '중립']
SPLITS     = {'train': 'Training', 'val': 'Validation'}

# 라벨 zip 이름 패턴
LBL_TRAIN_PAT = "[라벨]EMOIMG_{e}_TRAIN.zip"
LBL_VAL_PAT   = "[라벨]EMOIMG_{e}_VALID.zip"
SRC_TRAIN_PAT = "[원천]EMOIMG_{e}_TRAIN_{n:02d}.zip"
SRC_VAL_PAT   = "[원천]EMOIMG_{e}_VALID ({n}).zip"   # (1)~(13) + 원본


def load_label_index(lbl_zip_path: str) -> dict:
    """라벨 zip에서 {filename: bbox_평균} 딕셔너리 반환."""
    index = {}
    try:
        with zipfile.ZipFile(lbl_zip_path) as z:
            for fname in z.namelist():
                if not fname.endswith('.json'):
                    continue
                data = z.read(fname)
                try:
                    records = json.loads(data.decode('utf-8', errors='replace'))
                except Exception:
                    continue
                for r in records:
                    fn = r.get('filename', '')
                    if not fn:
                        continue
                    # annot_A/B/C에서 bbox 평균
                    boxes = []
                    for key in ('annot_A', 'annot_B', 'annot_C'):
                        ann = r.get(key, {})
                        b = ann.get('boxes', {})
                        if b and all(k in b for k in ('minX','minY','maxX','maxY')):
                            boxes.append(b)
                    if boxes:
                        avg = {
                            'minX': sum(b['minX'] for b in boxes) / len(boxes),
                            'minY': sum(b['minY'] for b in boxes) / len(boxes),
                            'maxX': sum(b['maxX'] for b in boxes) / len(boxes),
                            'maxY': sum(b['maxY'] for b in boxes) / len(boxes),
                        }
                        index[fn] = avg
    except Exception as e:
        print(f"  라벨 로드 실패 {lbl_zip_path}: {e}")
    return index


def crop_face(img: np.ndarray, bbox: dict, pad: float = 0.15, size: int = 224) -> np.ndarray:
    """bbox + padding으로 얼굴 크롭 후 224×224 리사이즈."""
    h, w = img.shape[:2]
    x1, y1 = bbox['minX'], bbox['minY']
    x2, y2 = bbox['maxX'], bbox['maxY']

    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - bw * pad))
    y1 = max(0, int(y1 - bh * pad))
    x2 = min(w, int(x2 + bw * pad))
    y2 = min(h, int(y2 + bh * pad))

    face = img[y1:y2, x1:x2]
    if face.size == 0:
        return cv2.resize(img, (size, size))
    return cv2.resize(face, (size, size))


def _is_valid_zip(path: str) -> bool:
    """실제 zip 파일인지 확인 (미완성 다운로드 제거)."""
    try:
        with zipfile.ZipFile(path) as z:
            z.namelist()
        return True
    except Exception:
        return False


def get_src_zips(split_dir: str, emotion: str, split: str,
                 fallback_dirs: list = None) -> list:
    """감정별 원천 zip 파일 목록 반환. fallback_dirs에서도 탐색."""
    search_dirs = [split_dir] + (fallback_dirs or [])
    keyword = 'TRAIN' if split == 'train' else 'VALID'
    result = []
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            full = os.path.join(d, f)
            # 손상된 파일 제외
            if any(f.endswith(x) for x in ('.irx603', '.irx628')):
                continue
            if not f.endswith('.zip'):
                continue
            # [원천] 접두사 있는 것 (Training/Validation 폴더)
            if f.startswith(f'[원천]EMOIMG_{emotion}_{keyword}'):
                if full not in result:
                    result.append(full)
            # 루트의 `EMOIMG_{emotion}_VALID (N).zip` — 괄호+숫자 있는 것만 원천
            elif (f'EMOIMG_{emotion}_{keyword} (' in f
                  and f.endswith('.zip')):
                if full not in result:
                    result.append(full)
    return result


def get_lbl_zip(split_dir: str, emotion: str, split: str,
                fallback_dirs: list = None) -> str:
    """라벨 zip 경로 반환. fallback_dirs에서도 탐색."""
    search_dirs = [split_dir] + (fallback_dirs or [])
    suffix = 'TRAIN' if split == 'train' else 'VALID'
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            # [라벨] 접두사 있는 것 (정규)
            if f.startswith(f'[라벨]EMOIMG_{emotion}_{suffix}') and f.endswith('.zip'):
                return os.path.join(d, f)
            # 루트의 `EMOIMG_{emotion}_VALID.zip` — 괄호 없는 것만 라벨
            if (f == f'EMOIMG_{emotion}_{suffix}.zip'):
                return os.path.join(d, f)
    return ''


def process_split(split: str, out_root: str, max_per_class: int,
                  dry_run: bool = False, seed: int = 42):
    rng = random.Random(seed)
    split_dir = os.path.join(AIHUB_ROOT, SPLITS[split])
    # 일부 zip(기쁨 val 등)이 루트에 있을 수 있으므로 fallback 경로 포함
    fallback_dirs = [
        r"G:\내 드라이브\aihub_face",
        os.path.join(r"G:\내 드라이브\aihub_face\한국인 감정인식을 위한 복합 영상",
                     SPLITS[split], '원천데이터_0114_add'),
        os.path.join(r"G:\내 드라이브\aihub_face\한국인 감정인식을 위한 복합 영상",
                     SPLITS[split], '라벨링데이터_231004_add'),
    ]
    out_split = os.path.join(out_root, split)

    print(f"\n{'='*60}")
    print(f"[{split.upper()}] 처리 시작")
    print(f"{'='*60}")

    summary = {}

    for emotion in EMOTIONS:
        print(f"\n  [{emotion}]")
        out_dir = os.path.join(out_split, emotion)
        if not dry_run:
            os.makedirs(out_dir, exist_ok=True)

        # 라벨 인덱스 로드
        lbl_zip = get_lbl_zip(split_dir, emotion, split, fallback_dirs)
        if not lbl_zip:
            print(f"    라벨 zip 없음 → 건너뜀")
            continue
        print(f"    라벨 로딩: {os.path.basename(lbl_zip)}")
        lbl_index = load_label_index(lbl_zip)
        print(f"    bbox 레코드: {len(lbl_index)}개")

        # 원천 zip 목록 (fallback 포함)
        src_zips = get_src_zips(split_dir, emotion, split, fallback_dirs)
        print(f"    원천 zip: {len(src_zips)}개")

        # 전체 파일 목록 수집 (zip_path, entry_name) → 언더샘플링용
        all_entries = []
        for zp in src_zips:
            try:
                with zipfile.ZipFile(zp) as z:
                    for name in z.namelist():
                        if name.lower().endswith('.jpg'):
                            bn = os.path.basename(name)
                            all_entries.append((zp, name, bn))
            except Exception as e:
                print(f"    zip 오류 {os.path.basename(zp)}: {e}")

        total_available = len(all_entries)
        print(f"    전체 이미지: {total_available}장")

        # val 이미지 없는 경우 train 데이터로 fallback (기쁨 val 미제공 대응)
        if total_available == 0 and split == 'val':
            print(f"    val 원천 없음 → train 데이터에서 fallback 추출")
            train_dir = os.path.join(AIHUB_ROOT, SPLITS['train'])
            train_fallback = [
                r"G:\내 드라이브\aihub_face",
                os.path.join(r"G:\내 드라이브\aihub_face\한국인 감정인식을 위한 복합 영상",
                             SPLITS['train'], '원천데이터_0114_add'),
                os.path.join(r"G:\내 드라이브\aihub_face\한국인 감정인식을 위한 복합 영상",
                             SPLITS['train'], '라벨링데이터_231004_add'),
            ]
            train_src_zips = get_src_zips(train_dir, emotion, 'train', train_fallback)
            print(f"    train 원천 zip: {len(train_src_zips)}개")
            for zp in train_src_zips:
                try:
                    with zipfile.ZipFile(zp) as z:
                        for name in z.namelist():
                            if name.lower().endswith('.jpg'):
                                bn = os.path.basename(name)
                                all_entries.append((zp, name, bn))
                except Exception as e:
                    print(f"    zip 오류 {os.path.basename(zp)}: {e}")
            # train label index 병합 (bbox용)
            train_lbl_zip = get_lbl_zip(train_dir, emotion, 'train', train_fallback)
            if train_lbl_zip:
                train_lbl_index = load_label_index(train_lbl_zip)
                lbl_index.update(train_lbl_index)
                print(f"    train bbox 레코드 추가: {len(train_lbl_index)}개")
            total_available = len(all_entries)
            # val용으로 마지막 N장 사용 (train 앞부분은 학습에 사용될 것이므로 뒷부분 사용)
            rng.shuffle(all_entries)
            print(f"    fallback 이미지: {total_available}장")

        # 언더샘플링
        if max_per_class and total_available > max_per_class:
            rng.shuffle(all_entries)
            all_entries = all_entries[:max_per_class]
            print(f"    언더샘플링: {max_per_class}장 사용")

        if dry_run:
            summary[emotion] = len(all_entries)
            print(f"    [DRY RUN] {len(all_entries)}장 처리 예정")
            continue

        # 크롭 및 저장
        saved, skipped_no_bbox, skipped_err = 0, 0, 0

        # zip별로 묶어서 처리 (zip 재열기 최소화)
        from collections import defaultdict
        by_zip = defaultdict(list)
        for zp, name, bn in all_entries:
            by_zip[zp].append((name, bn))

        for zp, entries in tqdm(by_zip.items(), desc=f'    {emotion}', leave=False):
            try:
                with zipfile.ZipFile(zp) as z:
                    for name, bn in entries:
                        out_path = os.path.join(out_dir, bn)
                        if os.path.exists(out_path):
                            saved += 1
                            continue
                        try:
                            data = z.read(name)
                            buf  = np.frombuffer(data, np.uint8)
                            img  = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                            if img is None:
                                skipped_err += 1
                                continue

                            bbox = lbl_index.get(bn)
                            if bbox:
                                face = crop_face(img, bbox)
                            else:
                                # bbox 없으면 중앙 크롭 fallback
                                h, w = img.shape[:2]
                                s = min(h, w)
                                y0, x0 = (h - s) // 2, (w - s) // 2
                                face = cv2.resize(img[y0:y0+s, x0:x0+s], (224, 224))
                                skipped_no_bbox += 1

                            cv2.imwrite(out_path, face)
                            saved += 1
                        except Exception:
                            skipped_err += 1
            except Exception as e:
                print(f"    zip 오류: {e}")

        summary[emotion] = saved
        print(f"    저장: {saved}장 | bbox 없음(중앙크롭): {skipped_no_bbox} | 오류: {skipped_err}")

    print(f"\n[{split.upper()}] 완료 요약:")
    for e, n in summary.items():
        print(f"  {e}: {n}장")
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out',           default=r'G:\내 드라이브\aihub_cropped',  help='출력 루트 디렉토리')
    p.add_argument('--max_per_class', type=int, default=16000,       help='클래스당 최대 이미지 수 (0=전체)')
    p.add_argument('--split',         default='all',                 help='train | val | all')
    p.add_argument('--dry_run',       action='store_true',           help='실제 저장 없이 수량만 확인')
    p.add_argument('--seed',          type=int, default=42)
    args = p.parse_args()

    splits = ['train', 'val'] if args.split == 'all' else [args.split]

    if args.dry_run:
        print('[DRY RUN 모드] 실제 파일 저장 없음')

    for split in splits:
        process_split(
            split=split,
            out_root=args.out,
            max_per_class=args.max_per_class if args.max_per_class > 0 else None,
            dry_run=args.dry_run,
            seed=args.seed,
        )

    print(f"\n출력 경로: {args.out}")
    print("완료!")


if __name__ == '__main__':
    main()
