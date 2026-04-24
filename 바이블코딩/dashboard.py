"""
바이블코딩 감정 데이터셋 분석 대시보드
실행: streamlit run dashboard.py
"""
import os
import json
import cv2
import numpy as np
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ── 경로 설정 ─────────────────────────────────────────────────────────
BASE       = os.path.dirname(os.path.abspath(__file__))
IMG_BASE   = os.path.join(BASE, '샘플데이터')
LBL_BASE   = os.path.join(BASE, '라벨링데이터')
OUTPUT_DIR = os.path.join(BASE, '..', 'output', 'bible')
EMOTIONS   = ['기쁨', '당황', '분노', '불안', '상처', '슬픔', '중립']
COLORS     = ['#f59e0b','#f97316','#ef4444','#8b5cf6','#6366f1','#3b82f6','#6b7280']
COLOR_MAP  = dict(zip(EMOTIONS, COLORS))

EXP_META = {
    'A_densenet121_ce':      {'label': 'A: DenseNet121 + CE',          'color': '#3b82f6'},
    'B_densenet121_focal':   {'label': 'B: DenseNet121 + Focal',        'color': '#22c55e'},
    'C_efficientnet_ce':     {'label': 'C: EfficientNet-B0 + CE',       'color': '#f59e0b'},
    'D_densenet121_focal_edge':{'label': 'D: DenseNet121 + Focal + Edge','color': '#ef4444'},
}

# ── 캐시된 데이터 로더 ────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_images(emotion, n=200):
    img_dir = os.path.join(IMG_BASE, emotion)
    files   = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg','.jpeg'))][:n]
    imgs = []
    for fn in files:
        with open(os.path.join(img_dir, fn), 'rb') as fh:
            buf = np.frombuffer(fh.read(), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is not None:
            imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return imgs

@st.cache_data(show_spinner=False)
def load_label(emotion):
    p = os.path.join(LBL_BASE, emotion, f'img_emotion_training_data({emotion}).json')
    with open(p, 'rb') as f:
        return json.loads(f.read())

@st.cache_data(show_spinner=False)
def compute_pixel_stats():
    rows = []
    for e in EMOTIONS:
        imgs = load_images(e)
        for img in imgs:
            arr  = img.astype(np.float32) / 255.
            hsv  = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
            sobel = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 1, ksize=3))
            rows.append({
                'emotion': e,
                'R': arr[:,:,0].mean(), 'G': arr[:,:,1].mean(), 'B': arr[:,:,2].mean(),
                'brightness': gray.mean(), 'contrast': gray.std(),
                'H': hsv[:,:,0].mean(), 'S': hsv[:,:,1].mean(), 'V': hsv[:,:,2].mean(),
                'edge_full':  sobel.mean(),
                'edge_brow':  sobel[20:55, 40:184].mean(),
                'edge_eye':   sobel[55:95, 40:184].mean(),
                'edge_mouth': sobel[130:175, 50:174].mean(),
                'asym_mouth': abs(gray[130:175,30:112].mean() - gray[130:175,112:194].mean()),
                'asym_eye':   abs(gray[55:95,20:112].mean()   - gray[55:95,112:204].mean()),
                'mouth_open': abs(gray[130:152,70:154].mean() - gray[152:175,70:154].mean()),
            })
    return pd.DataFrame(rows)

@st.cache_data(show_spinner=False)
def compute_haar_stats():
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    rows = []
    for e in EMOTIONS:
        imgs = load_images(e, n=100)
        for img in imgs:
            gray  = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(30,30))
            n_faces = len(faces) if len(faces) > 0 else 0
            area_ratio = 0.0
            if n_faces == 1:
                x,y,w,h = faces[0]
                area_ratio = (w*h) / (224*224)
            rows.append({'emotion': e, 'n_faces': n_faces, 'area_ratio': area_ratio})
    return pd.DataFrame(rows)

@st.cache_data(show_spinner=False)
def compute_demographics():
    rows = []
    for e in EMOTIONS:
        records = load_label(e)
        img_dir = os.path.join(IMG_BASE, e)
        existing = set(os.listdir(img_dir))
        for r in records:
            if r.get('filename','') not in existing:
                continue
            rows.append({'emotion': e, 'gender': r.get('gender','?'), 'age': r.get('age',-1)})
    return pd.DataFrame(rows)

@st.cache_data(show_spinner=False)
def compute_pca_features():
    rows, labels = [], []
    for e in EMOTIONS:
        imgs = load_images(e, n=100)
        for img in imgs:
            gray  = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
            hsv   = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
            sobel = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 1, ksize=3))
            rows.append([
                gray.mean(), gray.std(),
                hsv[:,:,0].mean(), hsv[:,:,1].mean(), hsv[:,:,2].mean(),
                gray[55:95,40:184].mean(), gray[130:175,50:174].mean(), gray[20:55,40:184].mean(),
                sobel.mean(), sobel[20:55].mean(), sobel[55:95].mean(), sobel[130:175].mean(),
                abs(gray[130:175,30:112].mean() - gray[130:175,112:194].mean()),
                abs(gray[55:95,20:112].mean()   - gray[55:95,112:204].mean()),
                abs(gray[130:152,70:154].mean() - gray[152:175,70:154].mean()),
            ])
            labels.append(e)
    X = StandardScaler().fit_transform(np.array(rows))
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)
    df = pd.DataFrame(X_pca, columns=['PC1','PC2','PC3'])
    df['emotion'] = labels
    return df, pca.explained_variance_ratio_

@st.cache_data(show_spinner=False)
def load_experiment_results():
    results = []
    if not os.path.isdir(OUTPUT_DIR):
        return results
    for exp_dir in sorted(Path(OUTPUT_DIR).iterdir()):
        rp = exp_dir / 'result.json'
        hp = exp_dir / 'history.json'
        if not rp.exists():
            continue
        with open(rp, encoding='utf-8') as f:
            r = json.load(f)
        history = []
        if hp.exists():
            with open(hp, encoding='utf-8') as f:
                history = json.load(f)
        results.append({'result': r, 'history': history, 'dir': str(exp_dir)})
    return results

# ── 모델 구조 시각화 ──────────────────────────────────────────────────
def draw_model_architecture(backbone: str, use_edge: bool, num_classes: int = 7):
    """Plotly로 모델 레이어 다이어그램 생성."""
    in_ch = 4 if use_edge else 3
    input_label = f'Input\n({in_ch}×224×224)'

    if backbone == 'densenet121':
        layers = [
            (input_label,        '#94a3b8', 0.6),
            ('Conv7×7 / MaxPool',  '#60a5fa', 0.8),
            ('Dense Block 1\n(6 layers)',   '#34d399', 1.2),
            ('Transition 1',       '#a78bfa', 0.5),
            ('Dense Block 2\n(12 layers)',  '#34d399', 1.6),
            ('Transition 2',       '#a78bfa', 0.5),
            ('Dense Block 3\n(24 layers)',  '#34d399', 2.0),
            ('Transition 3',       '#a78bfa', 0.5),
            ('Dense Block 4\n(16 layers)',  '#34d399', 1.8),
            ('BN + ReLU + GAP',    '#fb923c', 0.7),
            ('Dropout(0.3)',        '#f87171', 0.4),
            (f'FC → {num_classes}cls',      '#facc15', 0.6),
        ]
        title = f'DenseNet121 {"+ Edge(4ch)" if use_edge else "(3ch)"}'
    else:  # efficientnet_b0
        layers = [
            (input_label,         '#94a3b8', 0.6),
            ('Stem Conv3×3',       '#60a5fa', 0.7),
            ('MBConv1 ×1\n(32→16)', '#34d399', 1.0),
            ('MBConv6 ×2\n(16→24)', '#34d399', 1.2),
            ('MBConv6 ×2\n(24→40)', '#34d399', 1.2),
            ('MBConv6 ×3\n(40→80)', '#34d399', 1.4),
            ('MBConv6 ×3\n(80→112)','#34d399', 1.4),
            ('MBConv6 ×4\n(112→192)','#34d399', 1.6),
            ('MBConv6 ×1\n(192→320)','#34d399', 1.0),
            ('Conv1×1 + GAP',      '#fb923c', 0.7),
            ('Dropout(0.2)',        '#f87171', 0.4),
            (f'FC → {num_classes}cls',       '#facc15', 0.6),
        ]
        title = 'EfficientNet-B0 (3ch)'

    n = len(layers)
    x_center = 0.5
    y_positions = [1 - i / (n - 1) for i in range(n)]

    shapes, annotations = [], []
    for i, (label, color, height) in enumerate(layers):
        y = y_positions[i]
        w, h = 0.5, height * 0.06
        shapes.append(dict(
            type='rect',
            x0=x_center - w/2, x1=x_center + w/2,
            y0=y - h/2,         y1=y + h/2,
            fillcolor=color, opacity=0.85,
            line=dict(color='white', width=1.5),
        ))
        annotations.append(dict(
            x=x_center, y=y,
            text=label.replace('\n', '<br>'),
            showarrow=False,
            font=dict(size=11, color='white'),
            align='center',
        ))
        if i < n - 1:
            shapes.append(dict(
                type='line',
                x0=x_center, x1=x_center,
                y0=y - h/2,  y1=y_positions[i+1] + layers[i+1][2] * 0.06 / 2,
                line=dict(color='#64748b', width=1.5, dash='dot'),
            ))

    fig = go.Figure()
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color='#1e293b'), x=0.5),
        shapes=shapes,
        annotations=annotations,
        xaxis=dict(visible=False, range=[0, 1]),
        yaxis=dict(visible=False, range=[-0.05, 1.05]),
        height=600,
        margin=dict(l=10, r=10, t=50, b=10),
        plot_bgcolor='#f8fafc',
        paper_bgcolor='#f8fafc',
    )
    return fig


# ── 페이지 설정 ───────────────────────────────────────────────────────
st.set_page_config(page_title='감정인식 대시보드', layout='wide', page_icon='🧠')
st.title('🧠 감정인식 모델 대시보드')
st.caption('바이블코딩 데이터셋 분석 + 실험 결과 (7클래스 × 1,000장 = 7,000장)')

main_tab1, main_tab2, main_tab3 = st.tabs(['📊 데이터셋 분석', '🔬 실험 결과 & XAI', '📖 용어 사전'])


# ═══════════════════════════════════════════════════════════════════════
# TAB 1: 데이터셋 분석 (기존 내용)
# ═══════════════════════════════════════════════════════════════════════
with main_tab1:

    # 섹션 1: 개요
    st.header('1. 데이터셋 개요')
    col1, col2, col3, col4 = st.columns(4)
    col1.metric('총 이미지', '7,000장')
    col2.metric('클래스 수', '7개')
    col3.metric('이미지 크기', '224 × 224')
    col4.metric('레이블 일치율', '100%')

    fig_dist = go.Figure(go.Bar(
        x=EMOTIONS, y=[1000]*7,
        marker_color=COLORS, text=['1,000']*7, textposition='outside',
    ))
    fig_dist.update_layout(
        title='클래스별 이미지 수 (완전 균등)',
        yaxis=dict(range=[0,1200], title='이미지 수'),
        xaxis_title='감정 클래스', height=300, showlegend=False,
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    # 섹션 2: 픽셀 통계
    st.header('2. 픽셀 통계 분석')
    with st.spinner('픽셀 통계 계산 중...'):
        df_pixel = compute_pixel_stats()
    df_mean = df_pixel.groupby('emotion').mean().reindex(EMOTIONS)

    tab1, tab2, tab3 = st.tabs(['RGB 채널', '밝기 / 대비', 'HSV 색공간'])
    with tab1:
        fig_rgb = go.Figure()
        for ch, color in [('R','red'),('G','green'),('B','blue')]:
            fig_rgb.add_trace(go.Bar(name=ch, x=EMOTIONS, y=df_mean[ch], marker_color=color, opacity=0.7))
        fig_rgb.update_layout(barmode='group', title='클래스별 RGB 채널 평균', height=350)
        st.plotly_chart(fig_rgb, use_container_width=True)
        st.caption('클래스 간 채널 평균 차이 미미 → 조도 편향 없음')
    with tab2:
        fig_bc = go.Figure()
        fig_bc.add_trace(go.Bar(name='밝기(평균)', x=EMOTIONS, y=df_mean['brightness'],
                                marker_color=COLORS, opacity=0.85))
        fig_bc.add_trace(go.Scatter(name='대비(std)', x=EMOTIONS, y=df_mean['contrast'],
                                    mode='lines+markers', line=dict(color='black', width=2),
                                    yaxis='y2'))
        fig_bc.update_layout(
            title='클래스별 밝기 및 대비',
            yaxis=dict(title='밝기 (0~255)', range=[100,200]),
            yaxis2=dict(title='대비 (std)', overlaying='y', side='right', range=[0,60]),
            height=350,
        )
        st.plotly_chart(fig_bc, use_container_width=True)
    with tab3:
        fig_hsv = go.Figure()
        for ch, color in [('H','#ff6b35'),('S','#f7c59f'),('V','#efefd0')]:
            fig_hsv.add_trace(go.Bar(name=ch, x=EMOTIONS, y=df_mean[ch], marker_color=color, opacity=0.8))
        fig_hsv.update_layout(barmode='group', title='클래스별 HSV 평균', height=350)
        st.plotly_chart(fig_hsv, use_container_width=True)

    # 섹션 3: 얼굴 검출
    st.header('3. 얼굴 검출률 (Haar Cascade)')
    with st.spinner('얼굴 검출 중...'):
        df_haar = compute_haar_stats()
    df_haar_g = df_haar.groupby(['emotion','n_faces']).size().unstack(fill_value=0).reindex(EMOTIONS)
    detect_1     = df_haar_g.get(1, pd.Series(0, index=EMOTIONS))
    detect_multi = df_haar_g.get(2, pd.Series(0, index=EMOTIONS)) + df_haar_g.get(3, pd.Series(0, index=EMOTIONS))
    detect_none  = df_haar_g.get(0, pd.Series(0, index=EMOTIONS))
    fig_det = go.Figure()
    fig_det.add_trace(go.Bar(name='정확(1개)', x=EMOTIONS, y=detect_1,     marker_color='#22c55e'))
    fig_det.add_trace(go.Bar(name='다중',      x=EMOTIONS, y=detect_multi,  marker_color='#f59e0b'))
    fig_det.add_trace(go.Bar(name='미검출',    x=EMOTIONS, y=detect_none,   marker_color='#ef4444'))
    fig_det.update_layout(barmode='stack', title='클래스별 얼굴 검출 결과 (n=100)',
                          yaxis_title='이미지 수', height=380)
    st.plotly_chart(fig_det, use_container_width=True)
    st.warning('⚠️ **분노** 클래스 미검출률 66% — 눈썹 수축으로 Haar 특징점 실패 → 학습 시 전체 이미지 사용 권장')

    # 섹션 4: 영역별 특징
    st.header('4. 얼굴 영역별 특징 분석')
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader('엣지 밀도 (표현 강도)')
        fig_edge = go.Figure()
        for region, color in [('edge_brow','#7c3aed'),('edge_eye','#2563eb'),('edge_mouth','#dc2626')]:
            label = {'edge_brow':'눈썹','edge_eye':'눈','edge_mouth':'입'}[region]
            fig_edge.add_trace(go.Bar(name=label, x=EMOTIONS, y=df_mean[region], marker_color=color, opacity=0.8))
        fig_edge.update_layout(barmode='group', height=350, title='영역별 Sobel 엣지 밀도')
        st.plotly_chart(fig_edge, use_container_width=True)
    with col_b:
        st.subheader('비대칭성 & 입 열림')
        fig_asym = go.Figure()
        fig_asym.add_trace(go.Bar(name='입 비대칭', x=EMOTIONS, y=df_mean['asym_mouth'], marker_color='#f97316', opacity=0.8))
        fig_asym.add_trace(go.Bar(name='눈 비대칭', x=EMOTIONS, y=df_mean['asym_eye'],   marker_color='#0ea5e9', opacity=0.8))
        fig_asym.add_trace(go.Scatter(name='입 열림', x=EMOTIONS, y=df_mean['mouth_open'],
                                      mode='lines+markers', line=dict(color='black', width=2.5), yaxis='y2'))
        fig_asym.update_layout(barmode='group', height=350, title='비대칭성 & 입 열림 지표',
                               yaxis2=dict(title='입 열림', overlaying='y', side='right'))
        st.plotly_chart(fig_asym, use_container_width=True)

    # 섹션 5: 인구통계
    st.header('5. 인구통계 분포')
    with st.spinner('레이블 데이터 로딩 중...'):
        df_demo = compute_demographics()
    col_g, col_a2 = st.columns(2)
    with col_g:
        st.subheader('성별 분포')
        df_gender = df_demo.groupby(['emotion','gender']).size().unstack(fill_value=0).reindex(EMOTIONS)
        fig_g = go.Figure()
        for gender, color in [('남','#3b82f6'),('여','#ec4899')]:
            if gender in df_gender.columns:
                fig_g.add_trace(go.Bar(name=gender, x=EMOTIONS, y=df_gender[gender], marker_color=color, opacity=0.8))
        fig_g.update_layout(barmode='group', height=320, title='클래스별 성별 분포')
        st.plotly_chart(fig_g, use_container_width=True)
    with col_a2:
        st.subheader('연령대 분포')
        df_demo['age_group'] = pd.cut(df_demo['age'], bins=[0,19,29,39,49,99],
                                       labels=['10대','20대','30대','40대','50대+'])
        df_age = df_demo.groupby(['emotion','age_group']).size().unstack(fill_value=0).reindex(EMOTIONS)
        age_colors = ['#fde68a','#fbbf24','#f59e0b','#d97706','#92400e']
        fig_a = go.Figure()
        for i, ag in enumerate(['10대','20대','30대','40대','50대+']):
            if ag in df_age.columns:
                fig_a.add_trace(go.Bar(name=ag, x=EMOTIONS, y=df_age[ag],
                                       marker_color=age_colors[i], opacity=0.85))
        fig_a.update_layout(barmode='stack', height=320, title='클래스별 연령대 분포')
        st.plotly_chart(fig_a, use_container_width=True)

    # 섹션 6: PCA
    st.header('6. PCA 클래스 분리도')
    with st.spinner('PCA 계산 중...'):
        df_pca, var_ratio = compute_pca_features()
    col_p1, col_p2 = st.columns([2, 1])
    with col_p1:
        fig_pca = px.scatter(
            df_pca, x='PC1', y='PC2', color='emotion',
            color_discrete_map=COLOR_MAP, opacity=0.5,
            title=f'PCA 산점도 (PC1={var_ratio[0]:.1%}, PC2={var_ratio[1]:.1%})', height=450,
        )
        fig_pca.update_traces(marker=dict(size=5))
        st.plotly_chart(fig_pca, use_container_width=True)
    with col_p2:
        st.subheader('주성분 설명 분산')
        fig_var = go.Figure(go.Bar(
            x=[f'PC{i+1}' for i in range(len(var_ratio))], y=var_ratio,
            marker_color=['#4f46e5' if i < 3 else '#a5b4fc' for i in range(len(var_ratio))],
            text=[f'{v:.1%}' for v in var_ratio], textposition='outside',
        ))
        fig_var.update_layout(height=250, yaxis_title='설명 분산비', yaxis=dict(range=[0, 0.35]))
        st.plotly_chart(fig_var, use_container_width=True)
        df_center = df_pca.groupby('emotion')[['PC1','PC2']].mean().reindex(EMOTIONS).round(3)
        st.dataframe(df_center, height=280)
    st.info('💡 **분노**가 PC 공간에서 가장 고립됨 | **기쁨·당황·슬픔**은 중심 밀집 (혼동 가능)')

    # 섹션 7: 샘플 뷰어
    st.header('7. 샘플 이미지 뷰어')
    sel_emotion = st.selectbox('감정 선택', EMOTIONS)
    n_show = st.slider('표시 개수', 5, 20, 10)
    imgs = load_images(sel_emotion, n=n_show)
    cols = st.columns(5)
    for i, img in enumerate(imgs[:n_show]):
        cols[i % 5].image(img, use_column_width=True)

    # 섹션 8: 권장사항
    st.header('8. 학습 권장사항')
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.markdown("""
        **데이터 특성 요약**
        - ✅ 224×224 사전 크롭 완료
        - ✅ 클래스 균등 (1,000장씩)
        - ✅ 레이블 3인 완전 일치 (100%)
        - ✅ 손상 이미지 없음
        - ⚠️ 분노 Haar 검출 34% → 크롭 사용 금지
        - ⚠️ 20~30대/여성 편향
        """)
    with col_r2:
        st.markdown("""
        **권장 학습 설정**
        | 항목 | 권장값 |
        |---|---|
        | 전처리 | 리사이즈 불필요 |
        | 얼굴 크롭 | 사용 안 함 |
        | Class Weight | 불필요 |
        | Loss | Focal Loss 권장 |
        | 증강 | Flip + ColorJitter |
        | CLAHE | 불필요 (조도 균일) |
        """)


# ═══════════════════════════════════════════════════════════════════════
# TAB 2: 실험 결과 & XAI
# ═══════════════════════════════════════════════════════════════════════
with main_tab2:
    exp_data = load_experiment_results()

    if not exp_data:
        st.warning(f'실험 결과 없음. `{OUTPUT_DIR}` 경로를 확인하세요.')
        st.stop()

    results  = [d['result']  for d in exp_data]
    histories = [d['history'] for d in exp_data]
    exp_dirs  = [d['dir']     for d in exp_data]
    exp_names = [os.path.basename(d['dir']) for d in exp_data]

    # ── 2-1. 실험 개요 메트릭 ─────────────────────────────────────────
    st.header('실험 결과 비교')

    best_r = max(results, key=lambda r: r['test_f1'])
    best_name = os.path.basename(best_r['experiment'])

    m_cols = st.columns(4)
    m_cols[0].metric('최고 Test F1',  f"{best_r['test_f1']:.4f}", f"▲ Best: {best_name}")
    m_cols[1].metric('최고 Accuracy', f"{best_r['test_acc']:.2%}")
    m_cols[2].metric('ONNX 추론시간', f"{best_r['infer_onnx_ms']:.1f}ms", '≤2000ms 제약 충족')
    m_cols[3].metric('ONNX 크기',     f"{best_r['onnx_mb']:.1f}MB")

    # ── 2-2. 종합 비교표 ─────────────────────────────────────────────
    st.subheader('종합 비교표')
    rows_table = []
    for r in results:
        name = os.path.basename(r['experiment'])
        meta = EXP_META.get(name, {})
        rows_table.append({
            '실험':       meta.get('label', name),
            '백본':       r['backbone'],
            '손실함수':   r['loss'],
            '엣지채널':   'Yes' if r['use_edge'] else 'No',
            'Best Epoch': r['best_epoch'],
            'Val F1':     r['val_f1'],
            'Test F1':    r['test_f1'],
            'Test Acc':   f"{r['test_acc']:.2%}",
            'ONNX ms':    r['infer_onnx_ms'],
            'ONNX MB':    r['onnx_mb'],
            '추론 OK':    'OK' if r['infer_ok'] else 'FAIL',
        })
    df_table = pd.DataFrame(rows_table).sort_values('Test F1', ascending=False)
    st.dataframe(df_table, use_container_width=True, height=220)

    # ── 2-3. Macro F1 / Accuracy 비교 바 차트 ─────────────────────────
    st.subheader('Macro F1 & Accuracy 비교')
    sorted_results = sorted(results, key=lambda r: r['test_f1'], reverse=True)
    labels_r = [EXP_META.get(os.path.basename(r['experiment']), {}).get('label', os.path.basename(r['experiment']))
                for r in sorted_results]
    colors_r = [EXP_META.get(os.path.basename(r['experiment']), {}).get('color', '#94a3b8')
                for r in sorted_results]

    col_f1a, col_f1b = st.columns(2)
    with col_f1a:
        fig_f1 = go.Figure([
            go.Bar(name='Val F1',  x=labels_r, y=[r['val_f1']  for r in sorted_results],
                   marker_color='#93c5fd', opacity=0.8),
            go.Bar(name='Test F1', x=labels_r, y=[r['test_f1'] for r in sorted_results],
                   marker_color=colors_r, opacity=0.9),
        ])
        fig_f1.update_layout(barmode='group', title='Macro F1 비교', yaxis=dict(range=[0.8, 1.0]),
                             height=360, legend=dict(orientation='h', y=-0.2))
        st.plotly_chart(fig_f1, use_container_width=True)
    with col_f1b:
        fig_acc = go.Figure([
            go.Bar(name='Accuracy', x=labels_r,
                   y=[r['test_acc'] for r in sorted_results],
                   marker_color=colors_r, opacity=0.9,
                   text=[f"{r['test_acc']:.2%}" for r in sorted_results],
                   textposition='outside'),
        ])
        fig_acc.update_layout(title='Test Accuracy 비교',
                              yaxis=dict(range=[0.8, 1.0], tickformat='.0%'),
                              height=360, showlegend=False)
        st.plotly_chart(fig_acc, use_container_width=True)

    # ── 2-4. 클래스별 F1 히트맵 ───────────────────────────────────────
    st.subheader('클래스별 F1 히트맵')
    f1_matrix = []
    for r in sorted_results:
        row = [r['test_f1_per'].get(e, 0) for e in EMOTIONS]
        f1_matrix.append(row)

    exp_labels_short = [os.path.basename(r['experiment']) for r in sorted_results]
    fig_heat = go.Figure(go.Heatmap(
        z=f1_matrix,
        x=EMOTIONS,
        y=exp_labels_short,
        colorscale='RdYlGn',
        zmin=0.6, zmax=1.0,
        text=[[f'{v:.3f}' for v in row] for row in f1_matrix],
        texttemplate='%{text}',
        colorbar=dict(title='F1'),
    ))
    fig_heat.update_layout(title='클래스별 F1 (녹색=높음, 빨간=낮음)', height=300,
                           xaxis_title='감정 클래스', yaxis_title='실험')
    st.plotly_chart(fig_heat, use_container_width=True)
    st.info('💡 **불안** 클래스가 전 모델 공통 취약점 — Focal Loss(B) 적용 후 가장 많이 개선됨')

    # ── 2-5. 추론 속도 vs 성능 ────────────────────────────────────────
    st.subheader('추론 속도 vs 성능 (Trade-off)')
    fig_trade = go.Figure()
    for r in results:
        name = os.path.basename(r['experiment'])
        meta = EXP_META.get(name, {})
        fig_trade.add_trace(go.Scatter(
            x=[r['infer_onnx_ms']], y=[r['test_f1']],
            mode='markers+text',
            marker=dict(size=r['onnx_mb'] * 1.5, color=meta.get('color','#94a3b8'),
                        line=dict(color='white', width=2)),
            text=[meta.get('label', name)],
            textposition='top center',
            name=meta.get('label', name),
        ))
    fig_trade.add_vline(x=2000, line_dash='dash', line_color='red',
                        annotation_text='2000ms 제약')
    fig_trade.update_layout(
        title='ONNX 추론시간 vs Test F1 (버블 크기 = 모델 크기)',
        xaxis=dict(title='ONNX 추론시간 (ms)', range=[0, 100]),
        yaxis=dict(title='Test Macro F1', range=[0.84, 0.92]),
        height=400, showlegend=False,
    )
    st.plotly_chart(fig_trade, use_container_width=True)

    # ── 2-6. 학습 곡선 ────────────────────────────────────────────────
    st.subheader('학습 곡선')
    exp_sel = st.selectbox(
        '실험 선택',
        options=exp_names,
        format_func=lambda n: EXP_META.get(n, {}).get('label', n),
        key='history_sel',
    )
    sel_idx = exp_names.index(exp_sel)
    history = histories[sel_idx]

    if history:
        df_hist = pd.DataFrame(history)
        col_lc1, col_lc2 = st.columns(2)
        with col_lc1:
            fig_loss = go.Figure([
                go.Scatter(x=df_hist['epoch'], y=df_hist['train_loss'],
                           name='Train Loss', line=dict(color='#f97316', width=2)),
                go.Scatter(x=df_hist['epoch'], y=df_hist['val_loss'],
                           name='Val Loss', line=dict(color='#3b82f6', width=2, dash='dash')),
            ])
            fig_loss.update_layout(title='Loss 곡선', xaxis_title='Epoch',
                                   yaxis_title='Loss', height=320,
                                   legend=dict(orientation='h', y=-0.25))
            st.plotly_chart(fig_loss, use_container_width=True)
        with col_lc2:
            fig_metric = go.Figure([
                go.Scatter(x=df_hist['epoch'], y=df_hist['train_acc'],
                           name='Train Acc', line=dict(color='#f97316', width=2)),
                go.Scatter(x=df_hist['epoch'], y=df_hist['val_acc'],
                           name='Val Acc', line=dict(color='#3b82f6', width=2, dash='dash')),
                go.Scatter(x=df_hist['epoch'], y=df_hist['val_f1'],
                           name='Val F1', line=dict(color='#22c55e', width=2.5)),
            ])
            fig_metric.update_layout(title='Accuracy & F1 곡선', xaxis_title='Epoch',
                                     yaxis_title='Score', height=320,
                                     legend=dict(orientation='h', y=-0.25))
            st.plotly_chart(fig_metric, use_container_width=True)

        # 최고 지점 표시
        best_ep = df_hist.loc[df_hist['val_f1'].idxmax()]
        st.success(f"Best Epoch: **{int(best_ep['epoch'])}** — Val F1: {best_ep['val_f1']:.4f} | Val Acc: {best_ep['val_acc']:.4f}")
    else:
        st.info('history.json 없음')

    # ── 2-7. 혼동 행렬 ────────────────────────────────────────────────
    st.subheader('혼동 행렬')
    exp_cm_sel = st.selectbox(
        '실험 선택',
        options=exp_names,
        format_func=lambda n: EXP_META.get(n, {}).get('label', n),
        key='cm_sel',
    )
    cm_idx = exp_names.index(exp_cm_sel)
    cm = np.array(results[cm_idx]['confusion_matrix'])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    col_cm1, col_cm2 = st.columns(2)
    with col_cm1:
        fig_cm = go.Figure(go.Heatmap(
            z=cm, x=EMOTIONS, y=EMOTIONS,
            colorscale='Blues',
            text=cm, texttemplate='%{text}',
            colorbar=dict(title='Count'),
        ))
        fig_cm.update_layout(title='혼동 행렬 (절댓값)', height=420,
                             xaxis_title='예측', yaxis_title='정답')
        st.plotly_chart(fig_cm, use_container_width=True)
    with col_cm2:
        fig_cm_n = go.Figure(go.Heatmap(
            z=cm_norm, x=EMOTIONS, y=EMOTIONS,
            colorscale='RdYlGn', zmin=0, zmax=1,
            text=[[f'{v:.2f}' for v in row] for row in cm_norm],
            texttemplate='%{text}',
            colorbar=dict(title='Recall'),
        ))
        fig_cm_n.update_layout(title='혼동 행렬 (정규화, Recall)', height=420,
                               xaxis_title='예측', yaxis_title='정답')
        st.plotly_chart(fig_cm_n, use_container_width=True)

    # ── 2-8. GradCAM / XAI 이미지 뷰어 ───────────────────────────────
    st.subheader('GradCAM XAI 시각화')
    exp_xai_sel = st.selectbox(
        '실험 선택',
        options=exp_names,
        format_func=lambda n: EXP_META.get(n, {}).get('label', n),
        key='xai_sel',
    )
    xai_dir = os.path.join(exp_dirs[exp_names.index(exp_xai_sel)], 'xai')

    xai_tabs = st.tabs(['학습 곡선', '혼동 행렬', 'F1 레이더', 'GradCAM', '오분류 분석'])

    def show_png(path, caption=''):
        if os.path.exists(path):
            st.image(path, caption=caption, use_column_width=True)
        else:
            st.warning(f'파일 없음: {path}')

    with xai_tabs[0]:
        show_png(os.path.join(xai_dir, 'training_curves.png'), '학습 곡선')

    with xai_tabs[1]:
        show_png(os.path.join(xai_dir, 'confusion_matrix.png'), '혼동 행렬')

    with xai_tabs[2]:
        show_png(os.path.join(xai_dir, 'f1_radar.png'), 'F1 레이더 차트')

    with xai_tabs[3]:
        gradcam_dir = os.path.join(xai_dir, 'gradcam')
        if os.path.isdir(gradcam_dir):
            emotion_gcam = st.selectbox('감정 클래스 선택', EMOTIONS, key='gcam_emo')
            png_path = os.path.join(gradcam_dir, f'gradcam_{emotion_gcam}.png')
            show_png(png_path, f'{emotion_gcam} — GradCAM / GradCAM++ 비교')
        else:
            st.info('GradCAM 이미지 없음')

    with xai_tabs[4]:
        errors_dir = os.path.join(xai_dir, 'errors')
        if os.path.isdir(errors_dir):
            error_files = [f for f in os.listdir(errors_dir) if f.endswith('.png')]
            if error_files:
                for ef in sorted(error_files):
                    show_png(os.path.join(errors_dir, ef), ef.replace('.png', ''))
            else:
                st.info('오분류 이미지 없음')
        else:
            st.info('오분류 분석 결과 없음')

    # ── 2-9. 모델 구조 시각화 ─────────────────────────────────────────
    st.subheader('모델 구조 시각화')
    arch_cols = st.columns(len(exp_data))

    for i, (exp_dir, result) in enumerate(zip(exp_dirs, results)):
        name = exp_names[i]
        meta = EXP_META.get(name, {})
        backbone = result['backbone']
        use_edge = result['use_edge']
        with arch_cols[i]:
            st.markdown(f"**{meta.get('label', name)}**")
            st.plotly_chart(
                draw_model_architecture(backbone, use_edge),
                use_container_width=True,
                key=f'arch_{i}',
            )
            # 모델 통계
            st.markdown(f"""
            | 항목 | 값 |
            |---|---|
            | 백본 | `{backbone}` |
            | 입력 채널 | {'4 (RGB+Edge)' if use_edge else '3 (RGB)'} |
            | 손실함수 | `{result['loss']}` |
            | ONNX 크기 | {result['onnx_mb']}MB |
            | Test F1 | **{result['test_f1']:.4f}** |
            """)

    # ── 2-10. 결론 & 권장 모델 ────────────────────────────────────────
    st.subheader('결론 및 권장 모델')
    col_c1, col_c2 = st.columns([1, 1])
    with col_c1:
        st.success(f"""
        **Best 모델: {EXP_META.get(best_name, {}).get('label', best_name)}**
        - Test Macro F1: **{best_r['test_f1']:.4f}** ({best_r['test_acc']:.2%})
        - ONNX 추론시간: **{best_r['infer_onnx_ms']:.1f}ms** (≤2000ms)
        - ONNX 크기: **{best_r['onnx_mb']:.1f}MB**
        - FocalLoss(γ=2.0)가 CrossEntropy 대비 F1 +1.1% 향상
        """)
    with col_c2:
        worst_cls = min(best_r['test_f1_per'], key=best_r['test_f1_per'].get)
        st.warning(f"""
        **취약 클래스: {worst_cls}** (F1={best_r['test_f1_per'][worst_cls]:.4f})

        개선 방향:
        - 클래스 가중치 적용 (불안 클래스 weight 상향)
        - 불안 전용 데이터 증강 강화
        - Mixup / CutMix 적용 검토
        - Label Smoothing 조합
        """)

    st.markdown("""
    ---
    | 실험 | 특징 | 추천 용도 |
    |---|---|---|
    | **B** DenseNet121 + Focal | 최고 F1, 균형 잡힌 성능 | **프로덕션 기본** |
    | **C** EfficientNet + CE | 4배 빠른 추론, 크기 절반 | **엣지/모바일 배포** |
    | **A** DenseNet121 + CE | 안정적 베이스라인 | **비교 기준** |
    | **D** DenseNet121 + Edge | 엣지 채널 실험 (역효과) | **참고** |
    """)


# ═══════════════════════════════════════════════════════════════════════
# TAB 3: 용어 사전
# ═══════════════════════════════════════════════════════════════════════
with main_tab3:
    st.header('딥러닝 용어 사전')
    st.caption('본 프로젝트에서 사용된 주요 딥러닝 용어 정리 (활용도 순)')

    GLOSSARY = [
        ('딥러닝',        'Deep Learning',       '여러 층의 신경망으로 데이터에서 특징을 자동 학습하는 기계학습 기법',                          '"딥러닝 모델이 사람보다 빠르게 감정을 분류합니다"'),
        ('신경망',        'Neural Network',      '인간 뇌의 뉴런 구조를 모방한 수학적 모델',                                                  '"7층 신경망으로 이미지 분류를 수행합니다"'),
        ('합성곱 신경망', 'CNN',                 '이미지 처리에 특화된 신경망. 합성곱 레이어로 공간 특징 추출',                                 '"CNN 기반 DenseNet121로 얼굴 감정을 분류"'),
        ('감정인식',      'Emotion Recognition', '얼굴 이미지에서 감정 상태를 분류하는 기술',                                                  '"감정인식 모델이 기쁨을 94% 확률로 예측했습니다"'),
        ('추론',          'Inference',           '학습된 모델이 새 입력에 대해 예측값을 계산하는 과정',                                         '"추론 시간이 53ms로 2초 제약을 충족합니다"'),
        ('에폭',          'Epoch',               '전체 학습 데이터를 한 번 모두 학습하는 단위',                                                 '"30 에폭 학습 중 26번째에서 Best 달성"'),
        ('학습률',        'Learning Rate',       '가중치 업데이트 폭을 조절하는 하이퍼파라미터',                                               '"학습률 1e-4, AdamW 옵티마이저 사용"'),
        ('배치 크기',     'Batch Size',          '한 번의 가중치 업데이트에 사용되는 샘플 수',                                                  '"배치 크기 32로 학습"'),
        ('옵티마이저',    'Optimizer',           '손실을 줄이는 방향으로 가중치를 업데이트하는 알고리즘',                                       '"AdamW 옵티마이저로 빠른 수렴 달성"'),
        ('전이학습',      'Transfer Learning',   'ImageNet 등 대규모 데이터로 사전학습된 가중치를 재활용하는 기법',                             '"전이학습 덕분에 적은 데이터로도 높은 성능을 냈습니다"'),
        ('파인튜닝',      'Fine-tuning',         '사전학습 모델의 가중치를 새 데이터에 맞게 재학습하는 과정',                                   '"바이블코딩 데이터셋으로 파인튜닝했습니다"'),
        ('백본',          'Backbone',            '특징 추출을 담당하는 기반 신경망 구조',                                                      '"백본으로 DenseNet121을 사용했습니다"'),
        ('손실함수',      'Loss Function',       '모델 예측과 정답의 차이를 수치화하는 함수',                                                   '"손실함수로 Focal Loss를 사용했습니다"'),
        ('과적합',        'Overfitting',         '모델이 학습 데이터에 과도하게 맞춰져 새 데이터에서 성능이 떨어지는 현상',                     '"30 에폭 이후 val loss 증가 → 과적합 징후"'),
        ('F1 스코어',     'F1 Score',            '정밀도와 재현율의 조화평균. 불균형 데이터에 적합',                                            '"Macro F1=0.9046으로 Best 달성"'),
        ('혼동 행렬',     'Confusion Matrix',    '예측값과 실제값의 조합을 행렬로 표현한 성능 지표',                                            '"혼동 행렬에서 불안↔상처 혼동이 가장 빈번함"'),
        ('하이퍼파라미터','Hyperparameter',      '학습 전에 사람이 직접 설정하는 모델 외부 파라미터',                                           '"학습률·배치 크기·에폭 수가 핵심 하이퍼파라미터"'),
        ('정밀도',        'Precision',           '양성으로 예측한 것 중 실제 양성의 비율',                                                      '"기쁨 클래스 정밀도: 0.98"'),
        ('재현율',        'Recall',              '실제 양성 중 양성으로 예측한 비율',                                                           '"불안 클래스 재현율: 0.75"'),
        ('매크로 F1',     'Macro F1',            '클래스별 F1의 단순 평균. 모든 클래스를 동등하게 평가',                                        '"7개 클래스 Macro F1: 0.9046"'),
        ('오분류',        'Misclassification',   '모델이 실제 클래스와 다른 클래스로 예측한 경우',                                              '"불안→상처 오분류가 가장 빈번함"'),
        ('과소적합',      'Underfitting',        '모델이 학습 데이터조차 제대로 학습하지 못한 상태',                                            '"train acc가 60%에 머무르면 과소적합 의심"'),
        ('정규화',        'Regularization',      '과적합을 방지하기 위해 모델 복잡도에 제약을 가하는 기법',                                     '"Dropout과 Weight Decay로 정규화 적용"'),
        ('배치 정규화',   'Batch Normalization', '미니배치 단위로 레이어 입력을 정규화해 학습을 안정화하는 기법',                               '"Dense Block 내 각 레이어에 BN 적용"'),
        ('활성화 함수',   'Activation Function', '뉴런의 출력에 비선형성을 부여하는 함수',                                                      '"ReLU 활성화 함수로 음수 출력 제거"'),
        ('소프트맥스',    'Softmax',             '출력값을 0~1 확률 분포로 변환하는 함수. 다중 분류 최종 레이어에 사용',                        '"Softmax 출력에서 기쁨 확률 94% 반환"'),
        ('포컬 손실',     'Focal Loss',          '어려운 샘플(혼동 클래스)에 더 높은 가중치를 부여하는 손실함수',                              '"Focal Loss 적용 후 불안 클래스 F1 +2% 향상"'),
        ('데이터 증강',   'Data Augmentation',   '기존 이미지를 변환해 학습 데이터를 인위적으로 늘리는 기법',                                   '"RandomFlip, ColorJitter, RandomRotation 적용"'),
        ('드롭아웃',      'Dropout',             '학습 시 뉴런을 무작위로 비활성화하는 정규화 기법',                                            '"분류 레이어 앞에 Dropout(0.3) 적용"'),
        ('학습률 스케줄러','LR Scheduler',       '학습 진행에 따라 학습률을 자동으로 조정하는 기법',                                            '"CosineAnnealing으로 학습률을 점진적으로 감소"'),
        ('설명 가능한 AI','XAI',                 '모델의 예측 근거를 인간이 이해할 수 있게 시각화·설명하는 기술',                               '"XAI 파이프라인으로 모델 신뢰성 검증"'),
        ('GradCAM',       'GradCAM',             '그레이디언트를 활용해 모델이 어느 영역을 보고 예측했는지 시각화하는 XAI 기법',                '"GradCAM으로 기쁨 예측 시 입 주변에 집중함을 확인"'),
        ('특징 맵',       'Feature Map',         '합성곱 레이어를 통과한 후 생성되는 중간 표현. 각 채널이 특정 패턴을 감지',                    '"마지막 Dense Block의 특징 맵에 GradCAM 적용"'),
        ('앙상블',        'Ensemble',            '여러 모델의 예측을 결합해 단일 모델보다 높은 성능을 내는 기법',                               '"A·B 모델 앙상블 시 F1 +0.5% 추가 향상 가능"'),
        ('역전파',        'Backpropagation',     '그레이디언트를 출력층에서 입력층 방향으로 전파하는 학습 알고리즘',                            '"역전파로 계산된 그레이디언트로 GradCAM 생성"'),
        ('컨볼루션',      'Convolution',         '필터(커널)를 이미지 위에 슬라이딩하며 특징을 추출하는 연산',                                  '"3×3 컨볼루션 필터로 엣지·텍스처 특징 추출"'),
        ('풀링',          'Pooling',             '특징 맵의 공간 크기를 줄여 연산량을 감소시키는 레이어',                                       '"MaxPooling으로 특징 맵을 절반 크기로 축소"'),
        ('클래스 불균형', 'Class Imbalance',     '특정 클래스의 샘플 수가 다른 클래스보다 현저히 많거나 적은 상태',                             '"7개 클래스 균등 1,000장 → 클래스 불균형 없음"'),
        ('가중치 감쇠',   'Weight Decay',        '과적합 방지를 위해 가중치 크기에 페널티를 부여하는 정규화 기법',                              '"weight_decay=1e-4로 설정"'),
        ('전역 평균 풀링','GAP',                 '특징 맵 전체를 채널별 평균값 하나로 압축하는 레이어',                                         '"DenseNet121 마지막에 GAP 적용 후 FC 레이어 연결"'),
        ('주성분 분석',   'PCA',                 '고차원 데이터를 분산이 큰 방향으로 저차원으로 압축하는 기법',                                  '"PCA 산점도에서 분노 클래스가 가장 고립됨을 확인"'),
        ('ONNX',          'ONNX',                '프레임워크 간 모델 변환을 위한 표준 포맷. 서빙 최적화에 활용',                                '"PyTorch 모델을 ONNX로 변환 후 26.9MB, 추론 49ms"'),
        ('양자화',        'Quantization',        '모델 가중치를 32비트→8비트로 압축해 속도·크기를 줄이는 기법',                                '"INT8 양자화 적용 시 모델 크기 1/4로 감소"'),
        ('층화 분할',     'Stratified Split',    '클래스 비율을 유지하며 학습/검증/테스트 세트를 분리하는 방법',                                '"8:1:1 층화 분할로 각 세트에 클래스별 균등 분배"'),
        ('하르 캐스케이드','Haar Cascade',        '하르 특징과 AdaBoost로 얼굴 영역을 검출하는 전통적 알고리즘',                                '"분노 클래스 Haar 검출률 34% → 전체 이미지 학습으로 전환"'),
    ]

    df_glossary = pd.DataFrame(GLOSSARY, columns=['한글', '영어', '설명', '사용 예'])

    # 검색 필터
    search = st.text_input('🔍 용어 검색', placeholder='한글 또는 영어로 검색...')
    if search:
        mask = (df_glossary['한글'].str.contains(search, case=False, na=False) |
                df_glossary['영어'].str.contains(search, case=False, na=False) |
                df_glossary['설명'].str.contains(search, case=False, na=False))
        df_show = df_glossary[mask].reset_index(drop=True)
    else:
        df_show = df_glossary

    st.caption(f'총 {len(df_glossary)}개 용어 | 검색 결과 {len(df_show)}개')
    # 행 높이 35px 기준으로 전체 행이 보이도록 height 동적 계산
    row_height = 35
    header_height = 38
    st.dataframe(df_show, use_container_width=True,
                 height=header_height + row_height * len(df_show),
                 column_config={
                     '한글':  st.column_config.TextColumn('한글',  width=120),
                     '영어':  st.column_config.TextColumn('영어',  width=160),
                     '설명':  st.column_config.TextColumn('설명',  width=380),
                     '사용 예': st.column_config.TextColumn('사용 예', width=300),
                 })
