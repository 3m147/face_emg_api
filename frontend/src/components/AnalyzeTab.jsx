import { useState, useEffect, useRef } from 'react'
import { api } from '../api'

import { Card, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"

function ConfidenceBar({ emotion, score, highlight }) {
  return (
    <div className="flex items-center gap-3 mb-2">
      <span className="w-8 text-xs font-medium shrink-0 text-white/50">{emotion}</span>
      <div className="flex-1 h-2 rounded-full bg-white/[0.06] overflow-hidden">
        <div
          className="h-full rounded-full bg-white transition-all duration-700 ease-out"
          style={{ width: `${(score * 100).toFixed(1)}%`, opacity: highlight ? 1 : 0.25 }}
        />
      </div>
      <span className="w-12 text-[11px] text-right text-white/30 shrink-0 font-mono tabular-nums">
        {(score * 100).toFixed(1)}%
      </span>
    </div>
  )
}

function SingleResult({ result }) {
  const { emotion, confidence, scores, infer_ms } = result

  return (
    <Card className="glass animate-slide-up overflow-hidden">
      <CardContent className="p-0">
        <div className="text-center py-8 px-6">
          <div className="text-4xl font-black text-white tracking-tight mb-2">{emotion}</div>
          <div className="inline-flex items-center gap-2 bg-white/[0.06] rounded-full px-4 py-1.5 text-xs text-white/40">
            <span className="w-1.5 h-1.5 rounded-full bg-white/40 animate-pulse" />
            신뢰도 <span className="font-bold text-white">{(confidence * 100).toFixed(1)}%</span>
            {infer_ms && <span className="ml-1">· {infer_ms}ms</span>}
          </div>
        </div>
        <div className="px-5 pb-5 pt-2 border-t border-white/[0.04]">
          <h4 className="text-[10px] uppercase tracking-widest text-white/20 font-semibold mb-3">Scores</h4>
          {Object.keys(scores).map(e => (
            <ConfidenceBar key={e} emotion={e} score={scores[e] ?? 0} highlight={e === emotion} />
          ))}
        </div>
      </CardContent>
    </Card>
  )
}

function CompareResult({ results }) {
  return (
    <div className="space-y-3 animate-slide-up">
      <div className="flex items-center justify-between px-1 mb-1">
        <h3 className="text-sm font-bold text-white">비교 결과</h3>
        <Badge className="bg-white/10 text-white/50 border-white/10 text-[10px]">{results.length} Models</Badge>
      </div>
      {results.map((r, idx) => (
        <Card key={r.model_id} className="glass overflow-hidden animate-slide-up" style={{ animationDelay: `${idx * 80}ms` }}>
          <CardContent className="p-4">
            <div className="flex items-center justify-between mb-3">
              <div>
                <span className="font-semibold text-sm text-white">{r.model_label}</span>
                <div className="text-[10px] text-white/30 font-mono">{r.infer_ms}ms</div>
              </div>
              <span className="text-base font-black text-white">{r.emotion}</span>
            </div>
            <div className="flex items-center gap-2 mb-3 rounded-xl bg-white/[0.03] px-3 py-2">
              <span className="text-sm font-bold text-white/60">{r.emotion}</span>
              <div className="flex-1" />
              <span className="text-sm font-bold text-white font-mono">{(r.confidence * 100).toFixed(1)}%</span>
            </div>
            {Object.keys(r.scores).map(e => (
              <ConfidenceBar key={e} emotion={e} score={r.scores[e] ?? 0} highlight={e === r.emotion} />
            ))}
          </CardContent>
        </Card>
      ))}
    </div>
  )
}

export default function AnalyzeTab() {
  const [compareMode, setCompare] = useState(false)
  const [selectedModel, setModel] = useState('')
  const [availableModels, setAvailableModels] = useState([])
  const [preview, setPreview]     = useState(null)
  const [file, setFile]           = useState(null)
  const [loading, setLoading]     = useState(false)
  const [result, setResult]       = useState(null)
  const [error, setError]         = useState(null)
  const [faceB64, setFaceB64]     = useState(null)
  const [faceDetected, setFaceDetected] = useState(null)
  const [bbox, setBbox]           = useState(null)
  const [landmarks, setLandmarks] = useState([])
  const [imgSize, setImgSize]     = useState(null)

  const imgRef = useRef(null)
  const canvasRef = useRef(null)

  useEffect(() => {
    api.models().then(res => {
      const all = res.data.models || []
      setAvailableModels(all)
    }).catch(() => {})
  }, [])

  const onFileChange = (e) => {
    const f = e.target.files[0]
    if (!f) return
    setFile(f)
    setPreview(URL.createObjectURL(f))
    setResult(null)
    setError(null)
    setFaceB64(null)
    setBbox(null)
    setLandmarks([])
    setImgSize(null)
  }

  const analyze = async () => {
    if (!file) return
    setLoading(true)
    setError(null)
    setResult(null)
    setFaceB64(null)
    try {
      let res
      if (compareMode) {
        res = await api.analyzeCompare(file)
        setResult({ type: 'compare', data: res.data.results })
        setFaceB64(res.data.face_b64)
        setFaceDetected(res.data.face_detected)
        setBbox(res.data.bbox)
        if (res.data.landmarks) setLandmarks(res.data.landmarks)
      } else {
        res = await api.analyze(file, selectedModel)
        setResult({ type: 'single', data: res.data })
        setFaceB64(res.data.face_b64)
        setFaceDetected(res.data.face_detected)
        setBbox(res.data.bbox)
        if (res.data.landmarks) setLandmarks(res.data.landmarks)
      }
    } catch (e) {
      if (e?.code === 'ERR_NETWORK' || e?.message === 'Network Error') {
        setError('서버에 연결할 수 없습니다. 백엔드가 실행 중인지 확인하세요.')
      } else if (e?.response?.status === 503) {
        setError('모델이 아직 로딩 중입니다. 잠시 후 다시 시도하세요.')
      } else {
        const detail = e?.response?.data?.detail
        setError(detail ? `오류: ${detail}` : `서버 오류 (${e?.response?.status ?? 'unknown'})`)
      }
      console.error('[analyze error]', e)
    } finally {
      setLoading(false)
    }
  }

  // Draw overlay when results or image size are available
  useEffect(() => {
    if (!canvasRef.current || !imgSize) return
    const canvas = canvasRef.current
    const cw = canvas.offsetWidth
    const ch = canvas.offsetHeight
    if (!cw || !ch) return
    canvas.width = cw
    canvas.height = ch

    const ctx = canvas.getContext('2d')
    ctx.clearRect(0, 0, cw, ch)

    if (!result) return // Only draw over the image if we have a result

    const { w: imgW, h: imgH } = imgSize
    const scale = Math.max(cw / imgW, ch / imgH)
    const renderW = imgW * scale
    const renderH = imgH * scale
    const offsetX = (cw - renderW) / 2
    const offsetY = (ch - renderH) / 2

    // Unmirrored coordinate map for uploaded local image
    const mapPoint = (nx, ny) => {
      const rawX = offsetX + nx * renderW
      const rawY = offsetY + ny * renderH
      return [rawX, rawY]
    }

    if (bbox) {
      const [bx, by, bw, bh] = bbox
      const nx = bx / imgW
      const ny = by / imgH
      const nw = bw / imgW
      const nh = bh / imgH

      const [dx, dy] = mapPoint(nx, ny)
      const dw = nw * renderW
      const dh = nh * renderH

      ctx.strokeStyle = '#4A9EFF'
      ctx.lineWidth = 1.5
      ctx.setLineDash([6, 4])
      ctx.strokeRect(dx, dy, dw, dh)
      ctx.setLineDash([])

      const cl = 14
      ctx.lineWidth = 2.5
      ctx.strokeStyle = '#4A9EFF'
      ;[
        [dx, dy, 1, 1],
        [dx + dw, dy, -1, 1],
        [dx, dy + dh, 1, -1],
        [dx + dw, dy + dh, -1, -1],
      ].forEach(([cx2, cy2, sx2, sy2]) => {
        ctx.beginPath()
        ctx.moveTo(cx2 + sx2 * cl, cy2)
        ctx.lineTo(cx2, cy2)
        ctx.lineTo(cx2, cy2 + sy2 * cl)
        ctx.stroke()
      })
    }

    if (landmarks.length > 0) {
      landmarks.forEach(([nx, ny]) => {
        const [px, py] = mapPoint(nx, ny)
        ctx.beginPath()
        ctx.arc(px, py, 2.5, 0, Math.PI * 2)
        ctx.fillStyle = '#4ADE80'
        ctx.shadowColor = '#4ADE80'
        ctx.shadowBlur = 5
        ctx.fill()
      })
      ctx.shadowBlur = 0
    }
  }, [bbox, landmarks, imgSize, result])

  return (
    <div className="px-5 py-4 space-y-5">
      {/* Media Viewer */}
      <Card className="glass overflow-hidden border-0 glow-neon">
        <div className="aspect-[4/3] flex items-center justify-center relative">
          {preview ? (
            <>
              <img 
                ref={imgRef}
                src={preview} 
                alt="업로드" 
                className="w-full h-full object-cover" 
                onLoad={(e) => setImgSize({ w: e.target.naturalWidth, h: e.target.naturalHeight })}
              />
              <canvas
                ref={canvasRef}
                className="absolute inset-0 w-full h-full pointer-events-none"
              />
            </>
          ) : (
            <div className="flex flex-col items-center gap-3 text-muted-foreground/30">
              <div className="text-6xl">🖼️</div>
              <span className="text-xs font-medium tracking-wide">DROP IMAGE HERE</span>
            </div>
          )}
          
          {/* Overlay gradient */}
          {preview && (
            <div className="absolute inset-0 bg-gradient-to-t from-[#0a0a1a]/60 via-transparent to-transparent pointer-events-none" />
          )}
        </div>
      </Card>

      {/* File Upload */}
      <div>
        <input id="imgUpload" type="file" accept="image/*" className="hidden" onChange={onFileChange} />
        <Button asChild className={`w-full h-12 rounded-xl font-semibold ${preview ? 'glass border-white/10 text-muted-foreground hover:text-foreground' : 'bg-white/[0.06] border border-dashed border-white/10 text-muted-foreground hover:border-white/30 hover:text-white'}`}>
          <label htmlFor="imgUpload" className="cursor-pointer">
            {preview ? '🔄 다른 이미지 선택' : '📂 이미지 선택하기'}
          </label>
        </Button>
      </div>

      {/* Analysis Options */}
      <div className="space-y-3">
        <div className="glass rounded-2xl p-1 flex gap-1">
          <button
            className={`flex-1 py-2 text-xs font-semibold rounded-xl transition-all duration-300 ${!compareMode ? 'bg-white/10 text-white' : 'text-muted-foreground/50'}`}
            onClick={() => setCompare(false)}
          >단일 모델</button>
          <button
            className={`flex-1 py-2 text-xs font-semibold rounded-xl transition-all duration-300 ${compareMode ? 'bg-white/10 text-white' : 'text-muted-foreground/50'}`}
            onClick={() => setCompare(true)}
          >M-Ensemble 비교</button>
        </div>

        {!compareMode && (
          <select
            value={selectedModel}
            onChange={e => setModel(e.target.value)}
            className="w-full px-4 py-3 rounded-xl glass border-white/[0.08] text-sm font-medium text-foreground focus:outline-none focus:ring-1 focus:ring-white/20 appearance-none cursor-pointer"
          >
            <option value="" disabled hidden>선택해주세요</option>
            {availableModels.map(m => (
              <option key={m.id} value={m.id} disabled={!m.loaded}>
                {m.label}{!m.loaded ? ' (미로드)' : ''}
              </option>
            ))}
          </select>
        )}

        <Button
          className="w-full h-14 rounded-2xl font-bold text-base bg-white text-black hover:bg-white/90 transition-all duration-300 disabled:opacity-30 disabled:shadow-none"
          disabled={!file || loading || (!compareMode && !selectedModel)}
          onClick={analyze}
        >
          {loading ? (
            <span className="flex items-center gap-3">
              <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin"/>
              <span>추론 중...</span>
            </span>
          ) : '🚀 AI 분석 시작'}
        </Button>
        <p className="flex items-center justify-center gap-1.5 pt-1 pb-1 text-[10px] font-medium text-white/40">
          <svg className="w-3 h-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
          </svg>
          개인정보보호를 위해 데이터 수집은 하지 않습니다
        </p>
      </div>

      {/* Error */}
      {error && (
        <div className="animate-slide-up rounded-xl bg-red-500/10 border border-red-500/20 p-3">
          <p className="text-red-400 text-sm font-medium">{error}</p>
        </div>
      )}

      {/* Face Detection Status */}
      {faceDetected !== null && (
        <div className={`animate-fade-in rounded-xl p-3 text-xs font-medium border ${
          faceDetected 
            ? 'bg-emerald-500/10 border-emerald-500/15 text-emerald-400' 
            : 'bg-amber-500/10 border-amber-500/15 text-amber-400'
        }`}>
          {faceDetected ? '👀 얼굴 영역을 정확히 포착했습니다' : '⚠️ 얼굴 미검출 — 얼굴 개체를 찾을 수 없어 분석 결과를 표시하지 않습니다.'}
        </div>
      )}

      {/* Face Crop Preview */}
      {faceB64 && faceDetected && (
        <div className="animate-fade-in">
          <p className="text-[10px] font-semibold text-muted-foreground/50 mb-2 uppercase tracking-widest">Detected Region</p>
          <div className="w-16 h-16 rounded-2xl overflow-hidden border border-white/10 shadow-lg group">
            <img src={`data:image/jpeg;base64,${faceB64}`} alt="얼굴" className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-125" />
          </div>
        </div>
      )}

      {/* Results */}
      {result && faceDetected && (
        <div className="pt-2">
          {result.type === 'single'
            ? <SingleResult result={result.data} />
            : <CompareResult results={result.data} />}
        </div>
      )}
    </div>
  )
}
