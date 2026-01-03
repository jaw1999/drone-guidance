# Object Detection

## Overview

Object detection is handled by `ObjectDetector` in `src/core/detector.py`. It uses Ultralytics YOLO models exported to NCNN format for CPU inference on ARM processors.

## Models

Two model architectures are supported:

| Model | Parameters | Size | Notes |
|-------|------------|------|-------|
| YOLOv8n | 3.2M | 6.2 MB | Ultralytics YOLOv8 nano |
| YOLO11n | 2.6M | 5.4 MB | Ultralytics YOLO11 nano |

Both are trained on COCO dataset (80 classes).

## Resolutions

Models are exported at three input resolutions:

| Resolution | Inference (Pi 5) | Trade-off |
|------------|------------------|-----------|
| 640x640 | 110-130 ms | Detects smaller/distant objects |
| 416x416 | ~44 ms | Balanced |
| 320x320 | ~35 ms | Close-range only |

The resolution is the model input size. Input frames are resized (with letterboxing) to this size before inference. Output coordinates are scaled back to original frame dimensions.

## NCNN Backend

NCNN is Tencent's neural network inference framework optimized for mobile/embedded CPUs. The Ultralytics library handles NCNN loading when pointed to an exported model directory.

### Model Directory Structure

```
models/ncnn/yolov8n_640_ncnn_model/
├── model.ncnn.bin    # Weights
├── model.ncnn.param  # Network architecture
└── metadata.yaml     # Ultralytics metadata
```

### Thread Configuration

OpenMP threads are set before NumPy/NCNN imports:

```python
os.environ["OMP_NUM_THREADS"] = "3"
os.environ["OPENBLAS_NUM_THREADS"] = "3"
os.environ["MKL_NUM_THREADS"] = "3"
```

3 threads is optimal for Pi 5's quad-core Cortex-A76. Testing showed:
- 1 thread: ~150 ms
- 2 threads: ~115 ms
- 3 threads: ~110 ms
- 4 threads: ~115 ms (contention with other threads)

## Inference Pipeline

### 1. Model Loading

```python
from ultralytics import YOLO
model = YOLO("models/ncnn/yolov8n_640_ncnn_model")
```

Ultralytics detects the NCNN format from the directory structure.

### 2. Warmup

First inference initializes NCNN kernels and is slower. A dummy inference is run during initialization:

```python
dummy = np.zeros((640, 640, 3), dtype=np.uint8)
model.predict(dummy, imgsz=640, conf=0.5, verbose=False, device="cpu")
```

### 3. Detection

```python
results = model(
    frame,                                    # BGR numpy array
    imgsz=self.config.input_size,            # 640, 416, or 320
    conf=self.config.confidence_threshold,   # default 0.5
    iou=self.config.nms_threshold,           # default 0.45 (NMS IoU)
    verbose=False,
)
```

### 4. Result Parsing

Ultralytics returns a Results object. Boxes are extracted as:

```python
boxes = result.boxes.xyxy.cpu().numpy()      # (N, 4) - x1, y1, x2, y2
confs = result.boxes.conf.cpu().numpy()      # (N,) - confidence scores
class_ids = result.boxes.cls.cpu().numpy()   # (N,) - class indices
```

### 5. Class Filtering

If `target_classes` is specified in config, detections are filtered:

```python
mask = np.isin(class_ids, target_class_ids_array)
boxes, confs, class_ids = boxes[mask], confs[mask], class_ids[mask]
```

This uses numpy vectorized operations for speed.

## ROI Support

An optional region of interest can be passed to `detect()`:

```python
detections = detector.detect(frame, roi=(100, 100, 500, 400))
```

The frame is cropped before inference, and output coordinates are offset back to original frame space. This is not currently used in the main pipeline.

## Detection Dataclass

```python
@dataclass
class Detection:
    class_id: int           # COCO class index (0-79)
    class_name: str         # e.g., "person", "car"
    confidence: float       # 0.0 to 1.0
    bbox: tuple             # (x1, y1, x2, y2) in frame pixels
    center: tuple           # ((x1+x2)/2, (y1+y2)/2)

    @property
    def width(self) -> int

    @property
    def height(self) -> int

    @property
    def area(self) -> int
```

## Non-Maximum Suppression

NMS is handled by YOLO internally. The `nms_threshold` (config: `detector.nms_threshold`, default 0.45) is the IoU threshold. Boxes with IoU > threshold are suppressed, keeping the higher-confidence box.

## Error Handling

Detection errors are logged with progressive severity to avoid log spam:

| Consecutive Errors | Log Level |
|-------------------|-----------|
| 1 | WARNING |
| 2-10 | ERROR |
| 11+ | Suppressed (one CRITICAL message) |

Error count resets on successful detection.

## Configuration

```yaml
detector:
  model: "yolov8n"              # or "yolo11n"
  resolution: "640"             # "640", "416", or "320"
  confidence_threshold: 0.5     # minimum detection confidence
  nms_threshold: 0.45           # IoU threshold for NMS
  target_classes:               # empty = all classes
    - "person"
    - "car"
  detection_interval: 3         # run every N frames (used by Pipeline)
```

## COCO Classes

The 80 COCO classes (indices 0-79):

```
person, bicycle, car, motorcycle, airplane, bus, train, truck, boat,
traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat,
dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack,
umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball,
kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket,
bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple,
sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake,
chair, couch, potted plant, bed, dining table, toilet, tv, laptop,
mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink,
refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush
```

## Model Export

Models are exported using the script `scripts/export_ncnn.py`:

```bash
python scripts/export_ncnn.py yolov8n 640
```

This requires the `.pt` weights in `models/pt/` and runs:

```python
from ultralytics import YOLO
model = YOLO("models/pt/yolov8n.pt")
model.export(format="ncnn", imgsz=640, half=True)
```

The `half=True` flag enables FP16 weights which are faster on ARM.
