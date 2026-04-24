import { useState, useEffect, useRef, useCallback } from 'react'
import { api } from '../api'

const EMOTIONS = [
  { key: '기쁨',  label: '행복 (Happy)' },
  { key: '당황',  label: '당황 (Confused)' },
  { key: '분노',  label: '분노 (Angry)' },
  { key: '불안',  label: '불안 (Anxious)' },
  { key: '상처',  label: '상처 (Hurt)' },
  { key: '슬픔',  label: '슬픔 (Sad)' },
  { key: '중립',  label: '중립 (Neutral)' },
]
const ZERO_SCORES = EMOTIONS.reduce((a, e) => ({ ...a, [e.key]: 0 }), {})

// ── Canvas overlay drawing ───────────────────────────────────────────────────
function drawOverlay({ canvas, videoW, videoH, bbox, landmarks, showBbox, showLandmarks, faceId }) {
  if (!canvas || !videoW || !videoH) return
  const cw = canvas.offsetWidth
  const ch = canvas.offsetHeight
  if (!cw || !ch) return
  canvas.width  = cw
  canvas.height = ch

  const ctx = canvas.getContext('2d')
  ctx.clearRect(0, 0, cw, ch)

  // object-cover scaling metrics
  const scale = Math.max(cw / videoW, ch / videoH)
  const renderW = videoW * scale
  const renderH = videoH * scale
  const offsetX = (cw - renderW) / 2
  const offsetY = (ch - renderH) / 2

  // For a normalized x, y [0..1] map to mirrored canvas coordinates
  const mapPoint = (nx, ny) => {
    const rawX = offsetX + nx * renderW
    const rawY = offsetY + ny * renderH
    return [cw - rawX, rawY] // x is CSS-mirrored
  }

  if (showBbox && bbox) {
    const [bx, by, bw, bh] = bbox
    // Normalize bounding box coordinates
    const nx = bx / videoW
    const ny = by / videoH
    const nw = bw / videoW
    const nh = bh / videoH

    // Since image is mirrored, top-right of original box becomes top-left of mirrored box
    const [dx, dy] = mapPoint(nx + nw, ny)
    const dw = nw * renderW
    const dh = nh * renderH

    // Dashed border
    ctx.strokeStyle = '#4A9EFF'
    ctx.lineWidth   = 1.5
    ctx.setLineDash([6, 4])
    ctx.strokeRect(dx, dy, dw, dh)
    ctx.setLineDash([])

    // Corner accents
    const cl = 14
    ctx.lineWidth = 2.5
    ctx.strokeStyle = '#4A9EFF'
    ;[
      [dx,      dy,      1,  1],
      [dx + dw, dy,     -1,  1],
      [dx,      dy + dh, 1, -1],
      [dx + dw, dy + dh,-1, -1],
    ].forEach(([cx2, cy2, sx2, sy2]) => {
      ctx.beginPath()
      ctx.moveTo(cx2 + sx2 * cl, cy2)
      ctx.lineTo(cx2, cy2)
      ctx.lineTo(cx2, cy2 + sy2 * cl)
      ctx.stroke()
    })

    // FACE_ID label
    const label = `FACE_ID: ${faceId}`
    ctx.font = 'bold 11px monospace'
    const lw = ctx.measureText(label).width + 14
    const lh = 22
    const lx = dx
    const ly = Math.max(0, dy - lh - 6)
    ctx.fillStyle = 'rgba(0, 20, 60, 0.85)'
    ctx.fillRect(lx, ly, lw, lh)
    ctx.strokeStyle = '#4A9EFF'
    ctx.lineWidth = 1
    ctx.setLineDash([])
    ctx.strokeRect(lx, ly, lw, lh)
    ctx.fillStyle = '#ffffff'
    ctx.fillText(label, lx + 7, ly + 15)
  }

  if (showLandmarks && landmarks.length > 0) {
    landmarks.forEach(([nx, ny]) => {
      const [px, py] = mapPoint(nx, ny)
      ctx.beginPath()
      ctx.arc(px, py, 2.5, 0, Math.PI * 2)
      ctx.fillStyle = '#4ADE80'
      ctx.shadowColor = '#4ADE80'
      ctx.shadowBlur  = 5
      ctx.fill()
    })
    ctx.shadowBlur = 0
  }
}

// ── Toggle switch ────────────────────────────────────────────────────────────
function Toggle({ value, onChange }) {
  return (
    <button
      onClick={() => onChange(!value)}
      className={`relative w-9 h-5 rounded-full transition-colors duration-200 shrink-0 ${
        value ? 'bg-[#4A9EFF]' : 'bg-white/20'
      }`}
    >
      <div className={`absolute top-0.5 w-4 h-4 rounded-full bg-white shadow transition-all duration-200 ${
        value ? 'left-4' : 'left-0.5'
      }`} />
    </button>
  )
}

// ── Main component ───────────────────────────────────────────────────────────
export default function RealtimeTab() {
  const [cameraOn, setCameraOn]     = useState(false)
  const [scores, setScores]         = useState(ZERO_SCORES)
  const [inferMs, setInferMs]       = useState(null)
  const [confidence, setConfidence] = useState(null)
  const [topEmotion, setTopEmotion] = useState(null)
  const [error, setError]           = useState(null)
  const [selectedModel, setSelectedModel] = useState('')
  const [availableModels, setAvailableModels] = useState([])
  const [landmarks, setLandmarks]   = useState([])
  const [bbox, setBbox]             = useState(null)
  const [faceDetected, setFaceDetected] = useState(false)
  const [showBbox, setShowBbox]     = useState(true)
  const [showLandmarks, setShowLandmarks] = useState(true)
  const [showEmotions, setShowEmotions]   = useState(true)
  const [compareMode, setCompareMode]     = useState(false)
  const [compareResults, setCompareResults] = useState([])

  const faceIdRef      = useRef(Math.floor(Math.random() * 9000 + 1000).toString())
  const videoRef       = useRef(null)
  const canvasRef      = useRef(null)
  const streamRef      = useRef(null)
  const selectedModelRef = useRef('')
  const compareModeRef = useRef(false)
  const intervalRef    = useRef(null)
  const captureCanvas  = useRef(document.createElement('canvas'))
  const faceDetectedRef = useRef(false) // optional optimization, but we use state faceDetected
  const containerRef   = useRef(null)
  const processingRef  = useRef(false)
  const [isFullscreen, setIsFullscreen] = useState(false)

  // 풀스크린 상태 감지 (ESC 키 등으로 나갔을 때 동기화)
  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(!!document.fullscreenElement || !!document.webkitFullscreenElement)
    }
    document.addEventListener('fullscreenchange', handleFullscreenChange)
    document.addEventListener('webkitfullscreenchange', handleFullscreenChange)
    return () => {
      document.removeEventListener('fullscreenchange', handleFullscreenChange)
      document.removeEventListener('webkitfullscreenchange', handleFullscreenChange)
    }
  }, [])

  const toggleFullscreen = async () => {
    const el = containerRef.current
    if (!el) return

    if (!isFullscreen) {
      try {
        if (el.requestFullscreen) {
          await el.requestFullscreen()
        } else if (el.webkitRequestFullscreen) {
          await el.webkitRequestFullscreen()
        } else {
          setIsFullscreen(true) // iOS Fallback
        }
      } catch (e) {
        setIsFullscreen(true) // Fallback
      }
    } else {
      try {
        if (document.exitFullscreen && document.fullscreenElement) {
          await document.exitFullscreen()
        } else if (document.webkitExitFullscreen && document.webkitFullscreenElement) {
          await document.webkitExitFullscreen()
        } else {
          setIsFullscreen(false)
        }
      } catch (e) {
        setIsFullscreen(false)
      }
    }
  }

  // Load model list
  useEffect(() => {
    api.models().then(res => {
      const all = res.data.models || []
      setAvailableModels(all)
    }).catch(() => {})
  }, [])

  useEffect(() => { selectedModelRef.current = selectedModel }, [selectedModel])
  useEffect(() => { compareModeRef.current = compareMode }, [compareMode])

  // Redraw canvas when overlay data changes
  useEffect(() => {
    const video = videoRef.current
    drawOverlay({
      canvas:        canvasRef.current,
      videoW:        video?.videoWidth,
      videoH:        video?.videoHeight,
      bbox,
      landmarks,
      showBbox,
      showLandmarks,
      faceId:        faceIdRef.current,
    })
  }, [bbox, landmarks, showBbox, showLandmarks, cameraOn])

  // Capture frame → analyze
  const captureAndAnalyze = useCallback(async () => {
    const isCompare = compareModeRef.current
    if (processingRef.current || (!isCompare && !selectedModelRef.current)) return
    const video = videoRef.current
    if (!video || video.readyState < 2) return

    processingRef.current = true
    try {
      const c = captureCanvas.current
      c.width  = video.videoWidth
      c.height = video.videoHeight
      c.getContext('2d').drawImage(video, 0, 0)
      const imageB64 = c.toDataURL('image/jpeg', 0.7)

      const res  = await api.analyzeBase64(imageB64, isCompare ? '' : selectedModelRef.current, isCompare)
      const data = res.data

      setError(null)
      if (data.landmarks) setLandmarks(data.landmarks)
      if (data.bbox)      setBbox(data.bbox)
      setFaceDetected(!!data.face_detected)

      if (isCompare && data.results) {
        setCompareResults(data.results)
        
        let bestEmotion = null
        let bestConf = 0
        data.results.forEach(r => {
          if (r.confidence > bestConf) {
            bestConf = r.confidence
            bestEmotion = r.emotion
          }
        })
        if (bestEmotion) {
          setTopEmotion(bestEmotion)
          setConfidence(bestConf)
        }
      } else if (!isCompare && data.scores) {
        setScores(data.scores)
        setInferMs(data.infer_ms)

        const sorted = Object.entries(data.scores).sort((a, b) => b[1] - a[1])
        if (sorted.length > 0) {
          setTopEmotion(sorted[0][0])
          setConfidence(sorted[0][1])
        }
      }
    } catch (e) {
      if (e?.code === 'ERR_NETWORK') setError('백엔드 서버에 연결할 수 없습니다')
    } finally {
      processingRef.current = false
    }
  }, [])

  // Camera start
  const startCamera = useCallback(async () => {
    setError(null)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
      })
      streamRef.current = stream
      setCameraOn(true)
    } catch {
      setError('카메라 접근 권한이 필요합니다')
    }
  }, [])

  // Attach stream + start interval
  useEffect(() => {
    if (cameraOn && streamRef.current && videoRef.current) {
      videoRef.current.srcObject = streamRef.current
      videoRef.current.play().catch(() => {})
      intervalRef.current = setInterval(captureAndAnalyze, 1000)
    }
    return () => clearInterval(intervalRef.current)
  }, [cameraOn, captureAndAnalyze])

  // Camera stop
  const stopCamera = useCallback(() => {
    setCameraOn(false)
    clearInterval(intervalRef.current)
    streamRef.current?.getTracks().forEach(t => t.stop())
    streamRef.current = null
    setScores(ZERO_SCORES)
    setInferMs(null)
    setConfidence(null)
    setTopEmotion(null)
    setLandmarks([])
    setBbox(null)
    setFaceDetected(false)
    const cv = canvasRef.current
    if (cv) cv.getContext('2d').clearRect(0, 0, cv.width, cv.height)
  }, [])

  useEffect(() => () => {
    clearInterval(intervalRef.current)
    streamRef.current?.getTracks().forEach(t => t.stop())
  }, [])

  // Sorted top-3 emotions for overlay panel
  const top3 = [...EMOTIONS]
    .map(e => ({ ...e, score: scores[e.key] ?? 0 }))
    .sort((a, b) => b.score - a.score)
    .slice(0, 3)

  const statusText    = !cameraOn ? '대기 중' : faceDetected ? '분석 중' : '얼굴 탐색'
  const confidencePct = confidence != null ? (confidence * 100).toFixed(1) + '%' : '—'

  return (
    <div className="flex flex-col font-mono select-none">

      {/* ── Stats Header ── */}
      <div className="px-4 pt-4 pb-3 border-b border-white/[0.06] flex items-start justify-between">
        <div>
          <p className="text-[9px] font-bold text-white/25 uppercase tracking-widest mb-2">
            AI Vision Analysis Simulator
          </p>
          <div className="flex items-center gap-6">
            {[
              { label: '상태',    value: statusText },
              { label: '지연 시간', value: inferMs ? `${inferMs}ms` : '—' },
              { label: '신뢰도',   value: confidencePct },
            ].map(s => (
              <div key={s.label}>
                <div className="text-[9px] text-white/30 uppercase tracking-wider">{s.label}</div>
                <div className="text-sm font-bold text-white tabular-nums">{s.value}</div>
              </div>
            ))}
          </div>
        </div>
        {cameraOn && (
          <div className="flex items-center gap-1.5 pt-1">
            <div className="w-1.5 h-1.5 rounded-full bg-white animate-pulse" />
            <span className="text-[9px] text-white/40 tracking-widest">LIVE</span>
          </div>
        )}
      </div>

      {/* ── Video + Canvas overlay ── */}
      <div 
        ref={containerRef}
        className={`${isFullscreen ? 'fixed inset-0 z-[9999] w-full h-[100dvh]' : 'relative aspect-[4/3]'} bg-black overflow-hidden flex items-center justify-center`}
      >
        <video
          ref={videoRef}
          autoPlay playsInline muted
          className={`absolute inset-0 w-full h-full object-cover -scale-x-100 ${cameraOn ? '' : 'hidden'}`}
        />
        <canvas
          ref={canvasRef}
          className="absolute inset-0 w-full h-full pointer-events-none"
          style={{ display: cameraOn ? 'block' : 'none' }}
        />

        {/* Camera off placeholder */}
        {!cameraOn && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-3"
            style={{ background: 'repeating-linear-gradient(0deg, #111 0px, #111 3px, #0e0e0e 3px, #0e0e0e 6px)' }}>
            <svg className="w-10 h-10 text-white/10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.2">
              <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/>
              <circle cx="12" cy="13" r="4"/>
            </svg>
            <p className="text-[10px] text-white/20 tracking-widest uppercase">Camera Off</p>
          </div>
        )}

        {/* Scanline overlay */}
        <div className="absolute inset-0 pointer-events-none" style={{
          background: 'repeating-linear-gradient(0deg, transparent 0px, transparent 3px, rgba(0,0,0,0.05) 3px, rgba(0,0,0,0.05) 4px)',
        }} />

        {/* Emotion panel (top-right) */}
        {cameraOn && showEmotions && !compareMode && (
          <div className="absolute top-3 right-3 w-44 rounded-lg overflow-hidden border border-white/10"
            style={{ background: 'rgba(0,0,0,0.72)', backdropFilter: 'blur(8px)' }}>
            <div className="px-3 pt-2.5 pb-2 space-y-2">
              {top3.map((e, i) => (
                <div key={e.key}>
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-[10px] text-white/70">{e.label}</span>
                    <span className="text-[10px] font-bold text-white tabular-nums">
                      {Math.round(e.score * 100)}%
                    </span>
                  </div>
                  <div className="h-1 rounded-full bg-white/10 overflow-hidden">
                    <div
                      className="h-full rounded-full transition-all duration-700"
                      style={{
                        width: `${e.score * 100}%`,
                        background: i === 0 ? '#4A9EFF' : 'rgba(255,255,255,0.4)',
                      }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Compare Mode panel (top-right) */}
        {cameraOn && showEmotions && compareMode && compareResults.length > 0 && (
          <div className="absolute top-3 right-3 w-48 rounded-lg overflow-hidden border border-white/10"
            style={{ background: 'rgba(0,0,0,0.72)', backdropFilter: 'blur(8px)' }}>
            <div className="px-3 pt-2.5 pb-2 space-y-2">
              <div className="text-[9px] text-white/50 mb-1 border-b border-white/10 pb-1">M-Ensemble 비교 결과</div>
              {compareResults.map(r => (
                <div key={r.model_id} className="flex flex-col gap-0.5">
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] font-medium text-white/90">{r.model_label.split(' ')[0]}</span>
                    <span className="text-[9px] text-white/50 font-mono">{r.infer_ms}ms</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-[12px] font-bold text-white">{r.emotion}</span>
                    <span className="text-[10px] text-white/80 tabular-nums">{(r.confidence * 100).toFixed(1)}%</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Fullscreen Toggle Button */}
        <button
          onClick={toggleFullscreen}
          className="absolute bottom-3 right-3 p-2.5 rounded-lg bg-black/40 hover:bg-black/60 text-white/70 hover:text-white backdrop-blur border border-white/10 transition-all duration-200 z-50"
          title="Fullscreen"
        >
          {isFullscreen ? (
            <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M8 3v3a2 2 0 0 1-2 2H3m18 0h-3a2 2 0 0 1-2-2V3m0 18v-3a2 2 0 0 1 2-2h3M3 16h3a2 2 0 0 1 2 2v3"/>
            </svg>
          ) : (
            <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3"/>
            </svg>
          )}
        </button>
      </div>

      {/* ── Error ── */}
      {error && (
        <div className="mx-4 mt-3 rounded-lg bg-white/[0.04] border border-white/10 px-3 py-2.5">
          <p className="text-white/50 text-xs">{error}</p>
        </div>
      )}

      {/* ── Controls ── */}
      <div className="px-4 py-4 space-y-3">

        {/* Analysis Options */}
        <div className="glass rounded-2xl p-1 flex gap-1 mb-1">
          <button
            onClick={() => setCompareMode(false)}
            className={`flex-1 py-1.5 text-[10px] font-semibold rounded-xl transition-all duration-300 ${!compareMode ? 'bg-white/10 text-white' : 'text-muted-foreground/50'}`}
          >단일 모델</button>
          <button
            onClick={() => setCompareMode(true)}
            className={`flex-1 py-1.5 text-[10px] font-semibold rounded-xl transition-all duration-300 ${compareMode ? 'bg-white/10 text-white' : 'text-muted-foreground/50'}`}
          >M-Ensemble 비교</button>
        </div>

        {/* Model selector */}
        {!compareMode && (
        <div className="flex items-center gap-3">
          <span className="text-[9px] text-white/30 uppercase tracking-widest shrink-0">분석 모드</span>
          <div className="relative flex-1">
            <select
              value={selectedModel}
              onChange={e => { setSelectedModel(e.target.value); selectedModelRef.current = e.target.value }}
              className="w-full appearance-none bg-white/[0.04] border border-white/[0.08] rounded-lg px-3 py-2 text-xs font-medium text-white focus:outline-none focus:border-white/20 cursor-pointer pr-6"
            >
              <option value="" disabled hidden>선택해주세요</option>
              {availableModels.map(m => (
                <option key={m.id} value={m.id} disabled={!m.loaded}>
                  {m.label}{!m.loaded ? ' (미로드)' : ''}
                </option>
              ))}
            </select>
            <svg className="absolute right-2 top-1/2 -translate-y-1/2 w-3 h-3 text-white/30 pointer-events-none" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="m6 9 6 6 6-6"/>
            </svg>
          </div>
        </div>
        )}

        {/* Toggle row */}
        <div className="grid grid-cols-2 gap-2">
          {[
            { label: '얼굴 영역 (Face Box)',       val: showBbox,      set: setShowBbox },
            { label: '랜드마크 (Landmarks)',       val: showLandmarks, set: setShowLandmarks },
            { label: '감정 퍼센트 (Emotion %)',    val: showEmotions,  set: setShowEmotions },
          ].map(item => (
            <div
              key={item.label}
              className="flex items-center justify-between px-3 py-2 rounded-lg border border-white/[0.07] bg-white/[0.03] gap-2"
            >
              <span className="text-[9px] text-white/40 leading-tight">{item.label}</span>
              <Toggle value={item.val} onChange={item.set} />
            </div>
          ))}
        </div>

        {/* Camera button */}
        <button
          onClick={cameraOn ? stopCamera : startCamera}
          className={`w-full py-3 rounded-xl font-bold text-sm transition-all duration-200 flex items-center justify-center gap-2 ${
            cameraOn
              ? 'bg-white/[0.06] text-white/60 border border-white/10 hover:bg-white/10'
              : 'bg-white text-black hover:bg-white/90'
          }`}
        >
          <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
            <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/>
            <circle cx="12" cy="13" r="4"/>
          </svg>
          {cameraOn ? '카메라 끄기' : '카메라 켜기'}
        </button>
        <p className="flex items-center justify-center gap-1.5 pt-1 pb-1 text-[10px] font-medium text-white/40">
          <svg className="w-3 h-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
          </svg>
          개인정보보호를 위해 데이터 수집은 하지 않습니다
        </p>
      </div>
    </div>
  )
}
