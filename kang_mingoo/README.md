# kang_mingoo

## How to run
```bash
python experiment.py --config kang_mingoo/config.yaml
```

## What to change
Edit `config.yaml` — backbone, loss, lr, epochs, etc.

| Option | Values |
|--------|--------|
| `backbone` | `efficientnet_b0` \| `densenet121` \| `densenet169` \| `resnet18` \| `resnet50` |
| `loss` | `ce` \| `focal` |
| `focal_gamma` | e.g. `1.0`, `2.0`, `5.0` |
| `use_edge` | `true` \| `false` (adds Canny edge as 4th channel) |
| `scheduler` | `cosine` \| `step` \| `none` |

## Results
Saved to `kang_mingoo/results/` after training:
- `best_model.pth` — best checkpoint
- `result.json` — test accuracy, F1, per-class F1
- `confusion_matrix.png`
- `grad_cam/` — Grad-CAM visualizations
