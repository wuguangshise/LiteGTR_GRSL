# LiteGTR

## Usage

### Installation

```bash
pip install -r requirements.txt
```

Install `torch` / `torchvision` for your CUDA version first (https://pytorch.org). `ultralytics==8.4.171` is needed only for the YOLO26n-P2 + LiteGTR model.

### Released models

| Config | Weights | Model |
|---|---|---|
| `configs/litegtr.yaml` | `weights/litegtr.pt` | LiteGTR |
| `configs/litegtr_s.yaml` | `weights/litegtr_s.pt` | LiteGTR-S |
| `configs/yolo26n_p2_litegtr.yaml` | `weights/yolo26n_p2_litegtr.pt` | YOLO26n-P2 + LiteGTR |

All three are trained on VisDrone2019-DET train.

### Run on the example images

```bash
python tools/demo.py --config configs/litegtr.yaml --weights weights/litegtr.pt
python tools/demo.py --config configs/litegtr_s.yaml --weights weights/litegtr_s.pt
python tools/demo.py --config configs/yolo26n_p2_litegtr.yaml --weights weights/yolo26n_p2_litegtr.pt
```

The five images in `assets/demo/` are processed by default; use `--source` for your own image or folder. The drawn detections and `detections.txt` (original-image pixels) are written to `runs/demo/<config name>/`. Add `--device cpu` to run without a GPU.

### Evaluate on VisDrone2019-DET val

Expected layout:

```
data/VisDrone/
  VisDrone2019-DET-train/{images,annotations}
  VisDrone2019-DET-val/{images,annotations}
```

```bash
python tools/val.py --config configs/litegtr.yaml --weights weights/litegtr.pt --data-root data/VisDrone
```

Metrics are computed with the COCO API in original-image pixels, with at most 100 detections per image; AP_vt and AP_t follow the AI-TOD size ranges (2-8 and 8-16 pixels).

### Parameters, GFLOPs and latency

```bash
python tools/profile_model.py --config configs/litegtr.yaml
python tools/latency.py --config configs/litegtr.yaml
```

Latency: PyTorch FP16, batch 1, 640 x 640, forward pass to decoded boxes (no NMS), median of 300 runs.

### Receiver coverage and region coverage

```bash
python tools/receiver_coverage.py --data-root data/VisDrone
```

No model or GPU is needed; only the VisDrone2019-DET val annotations and images (for the image sizes) are read. The script prints

- the receiver coverage `eta(s)`: the share of small objects whose box in the 640 x 640 letterboxed input contains at least one receiving position of a level with stride `s` (its cell centres), the share with at least two, and the median number of such positions per box (`s` = 4, 8, 16 for P2, P3, P4). Small objects are COCO small by original-image area (< 32^2), without ignored regions and the "others" class; boxes are mapped with the letterbox scale and padding, a centre counts as inside when `x0 <= c < x1`, and the median includes boxes with no position;
- the region coverage `kappa`: the share of the cells of an 8 x 8 grid over the input that hold at least one region-anchor token, with the token budgets and grids of `--config` (default `configs/litegtr.yaml`).

Expected output on VisDrone2019-DET val:

```
26575 small objects in VisDrone2019-DET-val at 640 input
  eta( 4) =  94.3%   >=2 positions  85.2%   median positions per box 4   (finest receiving level P2)
  eta( 8) =  72.2%   >=2 positions  42.8%   median positions per box 1   (finest receiving level P3)
  eta(16) =  33.8%   >=2 positions   3.2%   median positions per box 0   (finest receiving level P4)
region-anchor coverage kappa on the 8x8 grid: 62.5%
```

### Training

```bash
python tools/train.py --config configs/litegtr.yaml --data-root data/VisDrone --name litegtr
```

Outputs go to `runs/train/<name>/`. Resume with `--resume runs/train/<name>/weights/last.pt`.

## Acknowledgments

**中文**

感谢您抽出时间阅读我们的论文并了解 LiteGTR。

**English**

Thank you for taking the time to read our paper and explore LiteGTR.

**Español**

Gracias por dedicar su tiempo a leer nuestro artículo y conocer LiteGTR.
