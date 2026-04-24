#!/usr/bin/env bash
# face_emg(개발) → face_emg_service 프론트엔드 동기화
# CustomModelTab.jsx는 로컬 전용이므로 제외
# App.jsx / api.js는 서비스 버전(3탭, 커스텀 API 없음)을 유지

set -e

SRC="$(dirname "$0")/frontend/src"
DST="$(dirname "$0")/../face_emg_service/frontend/src"

if [ ! -d "$DST" ]; then
  echo "❌ face_emg_service 폴더를 찾을 수 없습니다: $DST"
  exit 1
fi

echo "🔄 공용 컴포넌트 동기화 중..."

# 공용 컴포넌트 (CustomModelTab 제외)
for f in AnalyzeTab.jsx ModelCompareTab.jsx PipelineTab.jsx; do
  cp "$SRC/components/$f" "$DST/components/$f"
  echo "  ✔ components/$f"
done

# 스타일 (공통)
cp "$SRC/index.css" "$DST/index.css"
echo "  ✔ index.css"

cp "$SRC/main.jsx" "$DST/main.jsx"
echo "  ✔ main.jsx"

echo ""
echo "⚠️  App.jsx / api.js 는 서비스 버전 별도 유지 (수동 반영 필요)"
echo ""
echo "✅ 동기화 완료 → face_emg_service에서 git push 실행하세요"
