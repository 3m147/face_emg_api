import { useState, useEffect } from 'react'

/* ── 감정 레이블 & 색상 ──────────────────────────────────────── */
const EMOTIONS = [
  { key: '기쁨',  en: 'Happy',    color: '#FFD700', icon: '😊' },
  { key: '당황',  en: 'Confused', color: '#FF8C42', icon: '😳' },
  { key: '분노',  en: 'Angry',    color: '#FF4A4A', icon: '😡' },
  { key: '불안',  en: 'Anxious',  color: '#A78BFA', icon: '😰' },
  { key: '상처',  en: 'Hurt',     color: '#60A5FA', icon: '🥺' },
  { key: '슬픔',  en: 'Sad',      color: '#38BDF8', icon: '😢' },
  { key: '중립',  en: 'Neutral',  color: '#9CA3AF', icon: '😐' },
]

/* ── 기능 카드 데이터 ─────────────────────────────────────────── */
const FEATURES = [
  {
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="w-6 h-6">
        <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z" />
        <circle cx="12" cy="13" r="4" />
      </svg>
    ),
    title: '실시간 감정 인식',
    desc: '웹캠으로 얼굴을 감지하고 7가지 감정을 1초 주기로 실시간 분석합니다.',
    badge: 'LIVE',
    color: '#4A9EFF',
  },
  {
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="w-6 h-6">
        <circle cx="12" cy="12" r="3" />
        <path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42" />
      </svg>
    ),
    title: '이미지 분석',
    desc: '업로드한 사진에서 얼굴을 탐지하고 감정 확률 분포를 시각화합니다.',
    badge: 'ANALYZE',
    color: '#A78BFA',
  },
  {
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="w-6 h-6">
        <path d="M3 3v18h18" /><path d="m19 9-5 5-4-4-3 3" />
      </svg>
    ),
    title: 'M-Ensemble 비교',
    desc: 'DenseNet121, ResNet50 등 여러 모델의 예측을 동시에 비교·분석합니다.',
    badge: 'BENCHMARK',
    color: '#34D399',
  },
  {
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="w-6 h-6">
        <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" />
      </svg>
    ),
    title: '데이터 파이프라인',
    desc: 'GradCAM, t-SNE, 엣지 샘플 등 학습 파이프라인 시각화를 확인합니다.',
    badge: 'XAI',
    color: '#FB923C',
  },
]

/* ── 떠다니는 파티클 ─────────────────────────────────────────── */
function Particles() {
  const pts = Array.from({ length: 18 }, (_, i) => ({
    id: i,
    x: Math.random() * 100,
    y: Math.random() * 100,
    size: Math.random() * 2 + 0.5,
    delay: Math.random() * 6,
    dur: Math.random() * 8 + 6,
    opacity: Math.random() * 0.25 + 0.05,
  }))
  return (
    <div className="pointer-events-none absolute inset-0 overflow-hidden">
      {pts.map(p => (
        <div
          key={p.id}
          className="absolute rounded-full bg-white"
          style={{
            left: `${p.x}%`,
            top: `${p.y}%`,
            width: p.size,
            height: p.size,
            opacity: p.opacity,
            animation: `float ${p.dur}s ease-in-out ${p.delay}s infinite alternate`,
          }}
        />
      ))}
    </div>
  )
}

/* ── 감정 롤링 카운터 ─────────────────────────────────────────── */
function EmotionTicker() {
  const [idx, setIdx] = useState(0)
  const [visible, setVisible] = useState(true)

  useEffect(() => {
    const iv = setInterval(() => {
      setVisible(false)
      setTimeout(() => {
        setIdx(p => (p + 1) % EMOTIONS.length)
        setVisible(true)
      }, 350)
    }, 2200)
    return () => clearInterval(iv)
  }, [])

  const em = EMOTIONS[idx]
  return (
    <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full border border-white/10 bg-white/[0.04]"
      style={{ backdropFilter: 'blur(12px)' }}>
      <span
        className="text-lg leading-none transition-all duration-300"
        style={{ opacity: visible ? 1 : 0, transform: visible ? 'translateY(0)' : 'translateY(-8px)' }}
      >
        {em.icon}
      </span>
      <span
        className="text-xs font-bold tracking-widest transition-all duration-300"
        style={{
          color: em.color,
          opacity: visible ? 1 : 0,
          transform: visible ? 'translateY(0)' : 'translateY(-8px)',
        }}
      >
        {em.key.toUpperCase()} · {em.en.toUpperCase()}
      </span>
      <span className="w-1.5 h-1.5 rounded-full animate-pulse" style={{ background: em.color }} />
    </div>
  )
}

/* ── 메인 컴포넌트 ──────────────────────────────────────────────── */
export default function StartPage({ onEnter }) {
  const [entered, setEntered] = useState(false)

  const handleEnter = () => {
    setEntered(true)
    setTimeout(() => onEnter?.(), 500)
  }

  return (
    <div
      className="relative flex flex-col min-h-[100dvh] w-full max-w-[480px] mx-auto overflow-y-auto overflow-x-hidden hide-scrollbar"
      style={{
        background: '#080808',
        opacity: entered ? 0 : 1,
        transform: entered ? 'scale(0.97)' : 'scale(1)',
        transition: 'opacity 0.45s ease, transform 0.45s ease',
      }}
    >
      {/* ── 배경 그라디언트 글로우 ── */}
      <div className="pointer-events-none absolute inset-0">
        <div
          className="absolute -top-32 -left-32 w-80 h-80 rounded-full"
          style={{ background: 'radial-gradient(circle, rgba(74,158,255,0.12) 0%, transparent 70%)' }}
        />
        <div
          className="absolute top-1/3 -right-24 w-64 h-64 rounded-full"
          style={{ background: 'radial-gradient(circle, rgba(167,139,250,0.10) 0%, transparent 70%)' }}
        />
        <div
          className="absolute bottom-16 left-1/4 w-56 h-56 rounded-full"
          style={{ background: 'radial-gradient(circle, rgba(52,211,153,0.08) 0%, transparent 70%)' }}
        />
        {/* 스캔라인 오버레이 */}
        <div
          className="absolute inset-0"
          style={{
            background: 'repeating-linear-gradient(0deg, transparent 0px, transparent 3px, rgba(0,0,0,0.06) 3px, rgba(0,0,0,0.06) 4px)',
          }}
        />
      </div>

      <Particles />

      <div className="relative z-10 flex flex-col px-5 pt-12 pb-8 gap-10">

        {/* ── 헤더 ── */}
        <header className="text-center flex flex-col items-center gap-4">
          {/* 로고 */}
          <div className="relative w-20 h-20 flex items-center justify-center">
            {/* 회전 링 */}
            <div
              className="absolute inset-0 rounded-full border border-white/10"
              style={{ animation: 'spin-slow 14s linear infinite' }}
            />
            <div
              className="absolute inset-2 rounded-full border border-dashed border-white/[0.06]"
              style={{ animation: 'spin-slow 10s linear infinite reverse' }}
            />
            {/* 로고 코어 */}
            <div
              className="w-14 h-14 rounded-full flex items-center justify-center font-extrabold text-xl text-black"
              style={{ background: 'linear-gradient(135deg, #ffffff 0%, #c0c0c0 100%)' }}
            >
              FA
            </div>
            {/* 펄스 링 */}
            <div
              className="absolute inset-0 rounded-full border border-white/20"
              style={{ animation: 'pulseRing 2.5s ease-out infinite' }}
            />
          </div>

          {/* 앱 이름 */}
          <div>
            <p className="text-[10px] font-bold tracking-[0.3em] text-white/25 uppercase mb-2">
              AI Emotion Recognition System
            </p>
            <h1 className="text-4xl font-extrabold text-white tracking-tight leading-none">
              Face<span className="text-white/30">·</span>Actor
            </h1>
            <p className="text-[11px] text-white/30 mt-2 tracking-widest">
              Powered by DenseNet121 · MediaPipe · ONNX Runtime
            </p>
          </div>

          {/* 감정 롤링 뱃지 */}
          <EmotionTicker />
        </header>

        {/* ── 감정 레인보우 바 ── */}
        <div>
          <p className="text-[9px] text-white/25 uppercase tracking-widest mb-2 text-center">
            7 Emotion Classes
          </p>
          <div className="flex rounded-xl overflow-hidden h-1.5">
            {EMOTIONS.map(e => (
              <div
                key={e.key}
                className="flex-1 transition-all duration-700"
                style={{ background: e.color }}
              />
            ))}
          </div>
          <div className="flex mt-2">
            {EMOTIONS.map(e => (
              <div key={e.key} className="flex-1 flex flex-col items-center gap-0.5">
                <span className="text-base leading-none">{e.icon}</span>
                <span className="text-[8px] text-white/30">{e.key}</span>
              </div>
            ))}
          </div>
        </div>

        {/* ── 기능 카드 그리드 ── */}
        <div className="grid grid-cols-2 gap-2.5">
          {FEATURES.map((f, i) => (
            <div
              key={f.title}
              className="glass rounded-2xl p-4 flex flex-col gap-3 animate-slide-up"
              style={{ animationDelay: `${i * 80}ms`, animationFillMode: 'both' }}
            >
              <div className="flex items-start justify-between">
                <div className="w-10 h-10 rounded-xl flex items-center justify-center"
                  style={{ background: `${f.color}18`, color: f.color }}>
                  {f.icon}
                </div>
                <span
                  className="text-[8px] font-bold tracking-widest px-2 py-0.5 rounded-full border"
                  style={{ color: f.color, borderColor: `${f.color}40`, background: `${f.color}10` }}
                >
                  {f.badge}
                </span>
              </div>
              <div>
                <p className="text-xs font-bold text-white leading-tight mb-1">{f.title}</p>
                <p className="text-[10px] text-white/35 leading-relaxed">{f.desc}</p>
              </div>
            </div>
          ))}
        </div>

        {/* ── 기술 스택 배지 ── */}
        <div className="flex flex-col items-center gap-3">
          <p className="text-[9px] text-white/20 uppercase tracking-widest">Tech Stack</p>
          <div className="flex flex-wrap justify-center gap-2">
            {['DenseNet121', 'ResNet50', 'MediaPipe', 'ONNX Runtime', 'FastAPI', 'React + Vite'].map(t => (
              <span
                key={t}
                className="text-[9px] font-mono font-semibold px-2.5 py-1 rounded-full border border-white/[0.08] text-white/40"
                style={{ background: 'rgba(255,255,255,0.03)' }}
              >
                {t}
              </span>
            ))}
          </div>
        </div>

        {/* ── 시작 버튼 ── */}
        <div className="flex flex-col items-center gap-3 pt-2">
          <button
            onClick={handleEnter}
            className="relative w-full py-4 rounded-2xl font-bold text-base text-black overflow-hidden transition-all duration-200 active:scale-[0.97]"
            style={{ background: 'linear-gradient(135deg, #ffffff 0%, #e0e0e0 100%)' }}
          >
            {/* 반짝이 */}
            <span
              className="absolute inset-y-0 w-1/4 rotate-12 bg-white/40"
              style={{ left: '-30%', animation: 'shimmer-btn 3s ease-in-out infinite' }}
            />
            <span className="relative flex items-center justify-center gap-2">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" className="w-4 h-4">
                <polygon points="5 3 19 12 5 21 5 3" />
              </svg>
              시작하기
            </span>
          </button>
          <p className="flex items-center gap-1.5 text-[10px] text-white/25">
            <svg className="w-3 h-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
            </svg>
            영상·이미지 데이터는 서버에 저장되지 않습니다
          </p>
        </div>

        {/* ── 푸터 ── */}
        <footer className="text-center pb-2">
          <p className="text-[9px] text-white/15 tracking-wider">
            © 2025 Face-Actor · Korean AI Hub EMG Dataset · MIT License
          </p>
        </footer>
      </div>

      {/* ── 인라인 키프레임 ── */}
      <style>{`
        @keyframes float {
          0%   { transform: translateY(0px); }
          100% { transform: translateY(-18px); }
        }
        @keyframes spin-slow {
          from { transform: rotate(0deg); }
          to   { transform: rotate(360deg); }
        }
        @keyframes shimmer-btn {
          0%   { left: -30%; opacity: 0.4; }
          50%  { left: 120%; opacity: 0.6; }
          100% { left: 120%; opacity: 0; }
        }
      `}</style>
    </div>
  )
}
