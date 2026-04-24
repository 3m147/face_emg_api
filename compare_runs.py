"""
실험 결과 비교 리포트
Usage: python compare_runs.py --dir output/bible
"""
import argparse
import json
import os
from pathlib import Path


def load_results(root: str) -> list:
    results = []
    for p in sorted(Path(root).rglob('result.json')):
        with open(p, encoding='utf-8') as f:
            results.append(json.load(f))
    return results


def print_report(results: list):
    if not results:
        print('결과 없음')
        return

    emotions = list(results[0]['test_f1_per'].keys())

    # 헤더
    w = 30
    print(f"\n{'='*90}")
    print('실험 결과 비교 (test Macro F1 기준)')
    print(f"{'='*90}")
    header = f"{'실험':<{w}} {'val F1':>7} {'test F1':>8} {'acc':>6} {'ONNX ms':>9} {'MB':>5} {'OK':>3}"
    print(header)
    print('-'*90)

    sorted_r = sorted(results, key=lambda x: x['test_f1'], reverse=True)
    for r in sorted_r:
        name = os.path.basename(r['experiment'])
        ok   = 'OK' if r['infer_ok'] else 'NG'
        print(f"{name:<{w}} {r['val_f1']:>7.4f} {r['test_f1']:>8.4f} "
              f"{r['test_acc']:>6.4f} {r['infer_onnx_ms']:>9.1f} "
              f"{r['onnx_mb']:>5.1f} {ok:>3}")

    # 클래스별 F1 비교
    print(f"\n{'='*90}")
    print('클래스별 F1 (test)')
    print(f"{'='*90}")
    e_header = f"{'실험':<{w}} " + ' '.join(f"{e[:2]:>6}" for e in emotions)
    print(e_header)
    print('-'*90)
    for r in sorted_r:
        name = os.path.basename(r['experiment'])
        row  = f"{name:<{w}} "
        for e in emotions:
            row += f"{r['test_f1_per'].get(e, 0):>6.4f} "
        print(row)

    # Best 모델
    best = sorted_r[0]
    print(f"\n{'='*90}")
    print(f"Best: {os.path.basename(best['experiment'])}")
    print(f"  test Macro F1 = {best['test_f1']:.4f}")
    print(f"  test Accuracy = {best['test_acc']:.4f}")
    print(f"  ONNX 추론시간 = {best['infer_onnx_ms']:.1f}ms (제약 ≤2000ms: {'OK' if best['infer_ok'] else 'FAIL'})")
    print(f"  ONNX 크기     = {best['onnx_mb']:.1f}MB")
    print(f"  클래스별 F1   = {best['test_f1_per']}")

    # 취약 클래스
    worst_cls = min(best['test_f1_per'], key=best['test_f1_per'].get)
    print(f"  취약 클래스   = {worst_cls} (F1={best['test_f1_per'][worst_cls]:.4f})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dir', default='output/bible', help='결과 검색 루트 디렉토리')
    args = p.parse_args()
    results = load_results(args.dir)
    print(f"{len(results)}개 실험 결과 로드")
    print_report(results)


if __name__ == '__main__':
    main()
