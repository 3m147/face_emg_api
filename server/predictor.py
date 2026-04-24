from __future__ import annotations
"""
감정인식 멀티모델 추론 관리자.
output/ 하위 4개 학습 결과를 모두 로드해 단일/비교 추론 지원.
"""
import base64
import logging
import os
import sys
import time

import cv2
import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)

# ── 인라인 상수 및 유틸 (dataset.py의 PyTorch 의존성 제거) ──────────────────

EMOTIONS = ['기쁨', '당황', '분노', '불안', '상처', '슬픔', '중립']
ALL_EMOTIONS = EMOTIONS
SAMPLE_EMOTIONS = ['기쁨', '당황', '분노', '상처']


def apply_clahe(img_rgb: np.ndarray) -> np.ndarray:
    """CLAHE를 LAB L채널에 적용해 조도 불균형 보정."""
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def extract_edge(img_rgb: np.ndarray) -> np.ndarray:
    """Canny 엣지맵 반환 (H, W), uint8 0~255."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    return cv2.Canny(gray, 50, 150)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'
)

# ── MediaPipe FaceMesh (랜드마크용) ───────────────────────────────────────────
try:
    import mediapipe as mp
    _FACE_MESH = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    _MEDIAPIPE_OK = True
    logger.info('MediaPipe FaceMesh 초기화 완료')
except Exception as _mp_err:
    _MEDIAPIPE_OK = False
    _FACE_MESH = None
    logger.warning(f'MediaPipe 초기화 실패 (Haar fallback): {_mp_err}')

# 시각화용 희소 랜드마크 인덱스 (~24개)
# Face oval — 전체 36개 중 격점 선택
_LM_OVAL  = [10, 297, 284, 389, 454, 361, 397, 379, 400, 152, 176, 150, 172, 132, 234, 162, 54, 109]
# 눈 근사 / 코끝 / 입꼬리
_LM_EXTRA = [33, 263, 1, 61, 291]
_KEY_LM   = _LM_OVAL + _LM_EXTRA

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# 4클래스 모델용
EMOTIONS = SAMPLE_EMOTIONS  # ['기쁨', '당황', '분노', '상처']
# 7클래스 모델용
EMOTIONS_7 = ALL_EMOTIONS   # ['기쁨', '당황', '분노', '불안', '상처', '슬픔', '중립']

EMOTION_EMOJI = {
    '기쁨': '😄', '당황': '😳', '분노': '😡', '상처': '😢',
    '불안': '😨', '슬픔': '😢', '중립': '😐',
}

# ── 모델 레지스트리 ────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    'mobilenet_v2': {
        'label':       'MobileNet-V2 (한유승)',
        'description': '4개 감정 분류 · 경량 모바일 모델',
        'ckpt':        '한유승/han_yooseung.onnx',
        'color':       '#F59E0B',
        'val_acc':     0.876,
        'f1_per':      {e: 0.88 for e in EMOTIONS},
        'num_classes': 4,
        'emotions':    EMOTIONS,
        'backbone':    'mobilenet_v2',
    },
    'efficientnet_v2_s': {
        'label':       'EfficientNetV2-S (신희원)',
        'description': '7개 감정 분류 · Acc 91.4%',
        'ckpt':        '신희원/efficientnet_v2_s.onnx',
        'color':       '#EC4899',
        'val_acc':     0.914,
        'f1_per':      {e: 0.91 for e in EMOTIONS_7},
        'num_classes': 7,
        'emotions':    EMOTIONS_7,
        'backbone':    'efficientnet_v2_s',
    },
    'resnet18': {
        'label':       'ResNet-18 (강민구)',
        'description': '7개 감정 분류 · 실시간 추론 모델',
        'ckpt':        'kang_mingoo/resnet18.onnx',
        'color':       '#22C55E',
        'val_acc':     0.82,
        'f1_per':      {e: 0.80 for e in EMOTIONS_7},
        'num_classes': 7,
        'emotions':    EMOTIONS_7,
        'backbone':    'resnet18',
    },
    'densenet': {
        'label':       'DenseNet-121 (박상훈)',
        'description': '4개 감정 분류 · DenseNet-121',
        'ckpt':        'park_sanghun/densenet121.onnx',
        'color':       '#06B6D4',
        'val_acc':     0.904,
        'f1_per':      {e: 0.90 for e in EMOTIONS},
        'num_classes': 4,
        'emotions':    EMOTIONS,
        'backbone':    'densenet121',
    },
}

# 파이프라인 시각화 이미지
PIPELINE_IMAGES = {
    'edge_samples':    'output/viz/edge_samples.png',
    'gradcam_samples': 'output/viz/gradcam_samples.png',
    'class_gradcam':   'output/viz/class_gradcam.png',
    'tsne':            'output/viz/tsne.png',
    'comparison':      'output/comparison.png',
}


# ── 단일 모델 추론기 ──────────────────────────────────────────────────────────

class EmotionPredictor:
    def __init__(self, model_id: str):
        self.model_id    = model_id
        self.info        = MODEL_REGISTRY[model_id]
        self.ort_session = None
        self.use_clahe   = False
        self.use_edge    = False
        self.in_channels = 3
        # 모델별 감정 리스트 (7클래스 vs 4클래스)
        self.emotions    = self.info.get('emotions', EMOTIONS)

    def load(self) -> bool:
        ckpt_path = os.path.join(BASE_DIR, self.info['ckpt'])
        if not os.path.isfile(ckpt_path):
            logger.warning(f'[{self.model_id}] 체크포인트 없음: {ckpt_path}')
            return False
        try:
            self.ort_session = ort.InferenceSession(ckpt_path)
            logger.info(f'[{self.model_id}] ONNX 로드 완료')
            return True
        except Exception as e:
            logger.error(f'[{self.model_id}] 로드 실패: {e}')
            import traceback; traceback.print_exc()
            return False

    def predict(self, face_rgb: np.ndarray) -> dict:
        """face_rgb: (H, W, 3) uint8 → 감정 예측 결과."""
        face = cv2.resize(face_rgb, (224, 224))

        if self.use_clahe:
            face = apply_clahe(face)

        face_f   = face.astype(np.float32) / 255.0
        face_norm = (face_f - MEAN) / STD
        rgb_tensor = face_norm.transpose(2, 0, 1)  # (3, H, W)

        if self.use_edge:
            edge = extract_edge(face).astype(np.float32) / 255.0
            edge_t = np.expand_dims(edge, axis=0)
            tensor = np.concatenate([rgb_tensor, edge_t], axis=0)
        else:
            tensor = rgb_tensor

        tensor = np.expand_dims(tensor, axis=0)

        t0 = time.time()
        if self.ort_session is not None:
            input_name = self.ort_session.get_inputs()[0].name
            ort_inputs = {input_name: tensor}
            logits_np = self.ort_session.run(None, ort_inputs)[0]
            # Softmax
            e_x = np.exp(logits_np - np.max(logits_np, axis=1, keepdims=True))
            probs = (e_x / e_x.sum(axis=1, keepdims=True))[0]
        else:
            probs = np.zeros(len(self.emotions), dtype=np.float32)
        elapsed = (time.time() - t0) * 1000

        emo_list = self.emotions
        pred_idx = int(probs.argmax())
        
        scores_dict = {e: float(probs[i]) for i, e in enumerate(emo_list)}
        
        # 4클래스 모델의 경우 나머지 3개 감정을 0.0으로 패딩하여 7개 감정으로 일치시킴
        for e in EMOTIONS_7:
            if e not in scores_dict:
                scores_dict[e] = 0.0

        return {
            'emotion':    emo_list[pred_idx],
            'emoji':      EMOTION_EMOJI.get(emo_list[pred_idx], '🤔'),
            'confidence': float(probs[pred_idx]),
            'scores':     scores_dict,
            'infer_ms':   round(elapsed, 1),
            'num_classes': len(EMOTIONS_7),
        }


# ── 얼굴 검출 ─────────────────────────────────────────────────────────────────

def detect_and_crop(img_bgr: np.ndarray):
    """
    Haar Cascade로 가장 큰 얼굴 검출 + 10% 패딩 크롭.
    반환: (bbox_or_None, face_bgr, face_b64)
    """
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30)
    )

    if len(faces) == 0:
        # 얼굴 미검출 → 중앙 정사각형 크롭
        h, w = img_bgr.shape[:2]
        s = min(h, w)
        x1, y1 = (w - s) // 2, (h - s) // 2
        face_bgr = img_bgr[y1:y1 + s, x1:x1 + s]
        bbox = None
    else:
        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        pad_x = int(fw * 0.1)
        pad_y = int(fh * 0.1)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(img_bgr.shape[1], x + fw + pad_x)
        y2 = min(img_bgr.shape[0], y + fh + pad_y)
        face_bgr = img_bgr[y1:y2, x1:x2]
        bbox = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]

    _, buf = cv2.imencode('.jpg', face_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    face_b64 = base64.b64encode(buf).decode('utf-8')
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    return bbox, face_rgb, face_b64


# ── MediaPipe 기반 얼굴 검출 + 랜드마크 ─────────────────────────────────────

def detect_with_landmarks(img_bgr: np.ndarray):
    """
    MediaPipe FaceMesh로 얼굴 검출 + 희소 랜드마크 추출.
    Returns: (bbox, landmarks_norm, face_rgb, face_b64)
      bbox:           [x, y, w, h] in pixels | None
      landmarks_norm: [[x, y], ...] normalized 0-1 | []
    """
    h, w = img_bgr.shape[:2]

    if _MEDIAPIPE_OK and _FACE_MESH is not None:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        res = _FACE_MESH.process(img_rgb)

        if res.multi_face_landmarks:
            fl = res.multi_face_landmarks[0]

            # Face oval bbox
            oval_xs = [fl.landmark[i].x for i in _LM_OVAL]
            oval_ys = [fl.landmark[i].y for i in _LM_OVAL]
            x1 = max(0.0, min(oval_xs) - 0.02)
            y1 = max(0.0, min(oval_ys) - 0.03)
            x2 = min(1.0, max(oval_xs) + 0.02)
            y2 = min(1.0, max(oval_ys) + 0.02)

            bx  = int(x1 * w);  by  = int(y1 * h)
            bw_ = int((x2 - x1) * w); bh_ = int((y2 - y1) * h)
            bbox = [bx, by, bw_, bh_]

            landmarks = [[round(fl.landmark[i].x, 4), round(fl.landmark[i].y, 4)]
                         for i in _KEY_LM]

            face_bgr = img_bgr[by:by + bh_, bx:bx + bw_]
            if face_bgr.size == 0:
                face_bgr = img_bgr
        else:
            bbox = None
            landmarks = []
            s = min(h, w)
            cx, cy = (w - s) // 2, (h - s) // 2
            face_bgr = img_bgr[cy:cy + s, cx:cx + s]
    else:
        # MediaPipe 미사용 — Haar fallback
        bbox, face_rgb, face_b64 = detect_and_crop(img_bgr)
        return bbox, [], face_rgb, face_b64

    _, buf = cv2.imencode('.jpg', face_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    face_b64 = base64.b64encode(buf).decode('utf-8')
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    return bbox, landmarks, face_rgb, face_b64


# ── 멀티모델 관리자 ───────────────────────────────────────────────────────────

class ModelManager:
    def __init__(self):
        self.predictors: dict[str, EmotionPredictor] = {}

    def load_all(self):
        for mid in MODEL_REGISTRY:
            p = EmotionPredictor(mid)
            if p.load():
                self.predictors[mid] = p
        logger.info(f'로드된 모델: {list(self.predictors.keys())}')

    def available_models(self) -> list:
        result = []
        for mid, info in MODEL_REGISTRY.items():
            result.append({
                'id':          mid,
                'label':       info['label'],
                'description': info['description'],
                'color':       info['color'],
                'loaded':      mid in self.predictors,
                'val_acc':     info['val_acc'],
                'f1_per':      info['f1_per'],
            })
        return result

    def predict_one(self, model_id: str, face_rgb: np.ndarray) -> dict | None:
        if model_id not in self.predictors:
            return None
        return self.predictors[model_id].predict(face_rgb)

    def predict_all(self, face_rgb: np.ndarray) -> list:
        results = []
        for mid in MODEL_REGISTRY:   # 등록 순서 유지
            if mid not in self.predictors:
                continue
            res = self.predictors[mid].predict(face_rgb)
            res['model_id']    = mid
            res['model_label'] = MODEL_REGISTRY[mid]['label']
            res['color']       = MODEL_REGISTRY[mid]['color']
            results.append(res)
        return results
