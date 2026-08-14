# OMR Solution Recommendations
### PaperCreator — AI-Powered Optical Mark Recognition

> **Stack Decision:** OMRChecker + YOLO11n + EfficientNet-B0 + Vision LLM Fallback  
> **Target Accuracy:** 99%+ on scanner input, 95%+ on mobile phone photos  
> **Deployment:** Self-hosted VPS (CPU-only inference, no GPU required)

---

## Table of Contents

1. [Stack Overview](#1-stack-overview)
2. [Why This Stack](#2-why-this-stack)
3. [4-Stage Robust Pipeline](#3-4-stage-robust-pipeline)
4. [Stage 1 — Preprocessing (OMRChecker + OpenCV)](#4-stage-1--preprocessing-omrchecker--opencv)
5. [Stage 2 — Detection (YOLO11n)](#5-stage-2--detection-yolo11n)
6. [Stage 3 — Classification (EfficientNet-B0)](#6-stage-3--classification-efficientnet-b0)
7. [Stage 4 — Post-processing (Rules + DBSCAN)](#7-stage-4--post-processing-rules--dbscan)
8. [3-Layer Confidence Cascade](#8-3-layer-confidence-cascade)
9. [Template Config Design](#9-template-config-design)
10. [Hardware Requirements](#10-hardware-requirements)
11. [Training Plan](#11-training-plan)
12. [Integration with PaperCreator](#12-integration-with-papercreator)
13. [Accuracy Expectations](#13-accuracy-expectations)
14. [Recommended Roadmap](#14-recommended-roadmap)

---

## 1. Stack Overview

| Component | Role | Why Chosen |
|---|---|---|
| **OMRChecker** | Preprocessing + deskew + perspective warp | Battle-tested, MIT licensed, JSON-driven template config |
| **YOLO11n** | Locate bubble grid on phone/scanner image | 57 FPS on CPU, 22% fewer params than YOLOv8, best accuracy-speed ratio in 2025 |
| **EfficientNet-B0** | Classify each bubble filled/empty | 5.3M params, ~20MB, 21 FPS on CPU, ideal for VPS deployment |
| **Vision LLM** | Fallback for ambiguous marks (~3%) | Claude Sonnet API or Qwen2.5-VL-7B (offline) |
| **FastAPI** | Python microservice wrapping the pipeline | Callable from ASP.NET Core Web API |

---

## 2. Why This Stack

### YOLO11 — Validated Choice

YOLO11 was released in September 2024 and consistently ranks among the best models in 2025 benchmarks due to its **C3k2 block** and **C2PSA mechanisms**, achieving high accuracy with low computational overhead. Key benchmarks:

- Surpasses YOLOv8 in mean Average Precision (mAP) while using **22% fewer parameters**
- Increases inference speed by **up to 30% on CPUs** compared to predecessors
- YOLO11n (nano): **~17ms/frame, ~57 FPS on CPU** — critical for VPS without GPU
- Ranked first in accuracy and among first in speed and model size in 2025 comparative reviews

### EfficientNet-B0 — Validated Choice

- Only **5.3M parameters** vs 25.6M for ResNet-50 at similar accuracy
- Model size: **~20MB** — 5x smaller than ResNet-50
- CPU inference: **~21.51 FPS** — significantly faster than ResNet-50 at 14.66 FPS
- Compound scaling optimally balances depth, width, and resolution
- Transfer learning on focused 2-class problems (filled/empty) consistently reaches **95–98% accuracy**
- Proven deployable on resource-constrained environments including Jetson Nano-class hardware

### OMRChecker — Role Clarification

OMRChecker handles what it does well and steps aside for what it doesn't:

| OMRChecker does | Your additions cover |
|---|---|
| Deskew, warp, threshold | CNN classification |
| JSON template config | YOLO11n detection for phone photos |
| Batch folder processing | Vision LLM fallback |
| Visual debug output | Multi-fill conflict resolution |

> OMRChecker achieves ~100% accuracy on good document scans and ~90% on mobile images. The AI layers push mobile accuracy above 97%.

---

## 3. 4-Stage Robust Pipeline

```
Image Input (phone / scanner)
        │
        ▼
┌─────────────────────────────────┐
│  Stage 1: Preprocessing        │  ← OMRChecker + OpenCV
│  Deskew · Warp · Threshold     │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  Stage 2: Detection            │  ← YOLO11n
│  Locate bubble grid            │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  Stage 3: Classification       │  ← EfficientNet-B0 + LLM fallback
│  Filled / Empty per bubble     │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  Stage 4: Post-processing      │  ← Rules + DBSCAN
│  Conflict resolve · Flag       │
└────────────────┬────────────────┘
                 │
                 ▼
        Result JSON + Audit Trail
```

---

## 4. Stage 1 — Preprocessing (OMRChecker + OpenCV)

Handles deskewing, perspective correction, and adaptive thresholding. Critical for phone photos with uneven lighting or slight rotation.

```python
import cv2
import numpy as np

def preprocess_sheet(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Deskew using moments
    coords = np.column_stack(np.where(gray < 128))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(gray, M, (w, h),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)

    # Perspective correction via 4 corner markers
    blurred = cv2.GaussianBlur(rotated, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    markers = sorted(contours, key=cv2.contourArea, reverse=True)[:4]
    pts = np.float32([cv2.boundingRect(m)[:2] for m in markers])
    pts = sort_corners(pts)
    dst = np.float32([[0, 0], [1654, 0], [1654, 2339], [0, 2339]])  # A4 at 200dpi
    H = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(rotated, H, (1654, 2339))

    # Adaptive threshold handles uneven lighting (critical for phone photos)
    thresh = cv2.adaptiveThreshold(warped, 255,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 15, 4)
    return thresh
```

**Template design requirements for best preprocessing results:**

- Print **4 corner registration markers** (solid black squares, 15–20mm)
- Minimum bubble size: **8–10mm diameter**
- Minimum scan resolution: **300 DPI**
- Adaptive thresholding handles phone photos automatically

---

## 5. Stage 2 — Detection (YOLO11n)

Use YOLO11n for phone-captured images where rotation, shadows, or partial occlusion make grid-based detection unreliable. Use contour/grid detection for clean scanner input.

```python
def detect_bubbles_yolo(image_path: str, model_path: str = "yolo11n_omr.pt") -> list[dict]:
    from ultralytics import YOLO
    model = YOLO(model_path)
    results = model(image_path, conf=0.4)[0]
    bubbles = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        bubbles.append({
            "x": x1, "y": y1,
            "center": ((x1+x2)//2, (y1+y2)//2),
            "yolo_conf": float(box.conf[0])
        })
    return bubbles

def detect_bubbles_grid(thresh: np.ndarray, config: dict) -> list[dict]:
    """Grid-based detection. Fastest for clean scanner input."""
    bubbles = []
    sx = config["roi"]["start_x"]
    sy = config["roi"]["start_y"]
    col_gap = config["roi"]["col_gap"]
    row_gap = config["roi"]["row_gap"]
    bw = config["roi"]["bubble_w"]
    bh = config["roi"]["bubble_h"]

    for q in range(config["active_questions"]):
        for opt_idx in range(config["options_per_question"]):
            x = sx + opt_idx * col_gap
            y = sy + q * row_gap
            roi = thresh[y:y+bh, x:x+bw]
            bubbles.append({
                "question": q + 1,
                "option": chr(65 + opt_idx),
                "x": x, "y": y,
                "roi": roi,
                "center": (x + bw // 2, y + bh // 2)
            })
    return bubbles
```

**YOLO11 variant selection guide:**

| Variant | Size | CPU Speed | Use when |
|---|---|---|---|
| YOLO11n | 3.2 MB | 57 FPS | Default — good quality phone photos |
| YOLO11s | 11 MB | 42 FPS | Low quality / crumpled sheets |
| YOLO11m | ~21 MB | 37 FPS | Only if s still misses bubbles |

---

## 6. Stage 3 — Classification (EfficientNet-B0)

### Model Architecture

```python
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, Model

def build_omr_classifier():
    base = EfficientNetB0(
        input_shape=(32, 32, 3),
        include_top=False,
        weights='imagenet'
    )
    base.trainable = False  # freeze backbone initially

    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    output = layers.Dense(2, activation='softmax')(x)  # [empty, filled]

    model = Model(base.input, output)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
```

### Training

```python
def train_classifier():
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=5,        # handle slight rotation
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.1,
        brightness_range=[0.8, 1.2],  # handle phone lighting variation
        validation_split=0.2
    )

    train_gen = datagen.flow_from_directory(
        'training_data/',
        target_size=(32, 32),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )
    val_gen = datagen.flow_from_directory(
        'training_data/',
        target_size=(32, 32),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )

    model = build_omr_classifier()
    model.fit(train_gen, validation_data=val_gen, epochs=20)

    # Fine-tune: unfreeze last 20 layers
    model.layers[0].trainable = True
    for layer in model.layers[0].layers[:-20]:
        layer.trainable = False

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_gen, validation_data=val_gen, epochs=10)
    model.save('efficientnet_b0_omr.h5')
```

**Training data requirements:**

- Minimum: 500 filled + 500 empty samples per paper type
- Source: real exam sheets with verified answers
- Augmentation covers rotation, brightness, zoom automatically
- After 2–3 verified exam batches you'll have sufficient data

### Inference — 3-Layer Confidence Cascade

```python
def classify_bubble(roi: np.ndarray, cnn_model, q_num: int, option: str) -> dict:
    """Full cascade: OpenCV → EfficientNet-B0 → Vision LLM"""

    # Layer 1: OpenCV pixel density (handles ~85% of bubbles)
    total = roi.shape[0] * roi.shape[1]
    fill_ratio = cv2.countNonZero(roi) / total

    if fill_ratio > 0.55:
        return {"answer": "FILLED", "confidence": 0.97, "source": "opencv"}
    elif fill_ratio < 0.22:
        return {"answer": "EMPTY",  "confidence": 0.97, "source": "opencv"}

    # Layer 2: EfficientNet-B0 (handles ~12% ambiguous)
    img = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
    img = cv2.resize(img, (32, 32)).astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    probs = cnn_model.predict(img, verbose=0)[0]
    confidence = float(max(probs))
    label = "FILLED" if probs[1] > probs[0] else "EMPTY"

    if confidence >= 0.85:
        return {"answer": label, "confidence": confidence, "source": "efficientnet_b0"}

    # Layer 3: Vision LLM fallback (~3% truly ambiguous)
    return check_bubble_vision_llm(roi, q_num, option)
```

**Confidence threshold guide:**

| OpenCV fill_ratio | Action |
|---|---|
| > 0.55 | Confirmed FILLED — skip CNN |
| < 0.22 | Confirmed EMPTY — skip CNN |
| 0.22 – 0.55 | Ambiguous — send to EfficientNet-B0 |

| CNN softmax confidence | Action |
|---|---|
| ≥ 0.85 | Accept CNN result |
| < 0.85 | Escalate to Vision LLM |

---

## 7. Stage 4 — Post-processing (Rules + DBSCAN)

Resolves multi-fill conflicts, detects erasures, flags anomalous answer patterns, and clusters free-form bubble detections by question.

```python
from collections import defaultdict
from sklearn.cluster import DBSCAN
import numpy as np

def postprocess_answers(classified_bubbles: list[dict], config: dict) -> list[dict]:
    by_question = defaultdict(list)
    for b in classified_bubbles:
        by_question[b["question"]].append(b)

    final_answers = []
    for q_num, options in sorted(by_question.items()):
        filled = [o for o in options if o["answer"] == "FILLED"]
        result = resolve_question(q_num, options, filled)
        final_answers.append(result)
    return final_answers


def resolve_question(q_num: int, all_options: list, filled: list) -> dict:
    base = {
        "question": q_num,
        "all_fills": {o["option"]: round(o["fill_ratio"], 3) for o in all_options},
        "flagged": False,
        "flag_reason": None
    }

    if len(filled) == 0:
        return {**base, "answer": None, "confidence": 0.95, "status": "unanswered"}

    if len(filled) == 1:
        b = filled[0]
        return {**base, "answer": b["option"], "confidence": b["confidence"],
                "status": "clean", "source": b["source"]}

    # Multi-fill conflict resolution
    best = max(filled, key=lambda x: x["fill_ratio"])
    second = sorted(filled, key=lambda x: x["fill_ratio"], reverse=True)[1]
    gap = best["fill_ratio"] - second["fill_ratio"]

    if gap < 0.12:
        return {**base, "answer": best["option"], "confidence": 0.55,
                "status": "multi_fill_ambiguous", "flagged": True,
                "flag_reason": f"Two bubbles close in density (gap={gap:.2f})"}

    if second["fill_ratio"] < 0.38:
        return {**base, "answer": best["option"], "confidence": 0.85,
                "status": "erasure_detected", "flagged": True,
                "flag_reason": "Possible erasure of previous answer"}

    return {**base, "answer": best["option"], "confidence": 0.72,
            "status": "multi_fill_resolved", "flagged": True,
            "flag_reason": f"{len(filled)} bubbles filled — picked darkest"}


def dbscan_cluster_free_form(bubbles: list[dict]) -> dict:
    """For YOLO-detected bubbles without grid config."""
    centers = np.array([b["center"] for b in bubbles])
    y_coords = centers[:, 1].reshape(-1, 1)
    db = DBSCAN(eps=20, min_samples=2).fit(y_coords)

    clustered = {}
    for idx, label in enumerate(db.labels_):
        if label == -1:
            continue
        clustered.setdefault(label, []).append(bubbles[idx])

    sorted_clusters = sorted(clustered.items(),
                              key=lambda kv: np.mean([b["center"][1] for b in kv[1]]))
    return {i+1: cluster for i, (_, cluster) in enumerate(sorted_clusters)}


def detect_sheet_anomalies(final_answers: list[dict]) -> list[str]:
    warnings = []
    answered = [a for a in final_answers if a["answer"] is not None]
    answers = [a["answer"] for a in answered]

    if len(set(answers)) == 1 and len(answers) > 5:
        warnings.append(f"All answers are '{answers[0]}' — possible marking error")

    flagged = [a for a in final_answers if a.get("flagged")]
    if len(flagged) / max(len(final_answers), 1) > 0.30:
        warnings.append(f"{len(flagged)} questions flagged — sheet quality may be poor")

    unanswered = [a for a in final_answers if a["answer"] is None]
    if len(unanswered) / max(len(final_answers), 1) > 0.15:
        warnings.append(f"{len(unanswered)} questions unanswered — verify sheet")

    return warnings
```

---

## 8. 3-Layer Confidence Cascade

```
Bubble ROI
    │
    ▼
OpenCV pixel density
    ├── fill_ratio > 0.55  ──→  FILLED (confidence 0.97)  ──→  Result
    ├── fill_ratio < 0.22  ──→  EMPTY  (confidence 0.97)  ──→  Result
    └── 0.22–0.55 (ambiguous)
              │
              ▼
        EfficientNet-B0
              ├── softmax ≥ 0.85  ──→  CNN result  ──→  Result
              └── softmax < 0.85
                        │
                        ▼
                  Vision LLM Fallback
                  (Claude Sonnet or Qwen2.5-VL-7B offline)
                        │
                        ▼
                  Final result + logged for CNN retraining
```

**Expected distribution:**
- OpenCV handles: ~85% of all bubbles
- EfficientNet-B0 handles: ~12% of all bubbles
- Vision LLM handles: ~3% of all bubbles

---

## 9. Template Config Design

One physical template, variable question counts — driven entirely by JSON config per exam:

```json
{
  "template": "standard_a4.png",
  "total_bubble_rows": 100,
  "active_questions": 50,
  "options_per_question": 4,
  "roi_grid": {
    "start_x": 120,
    "start_y": 180,
    "bubble_w": 32,
    "bubble_h": 32,
    "col_gap": 48,
    "row_gap": 36
  },
  "answer_key": {
    "1": "A",
    "2": "C",
    "3": "B"
  },
  "scoring": {
    "marks_correct": 1,
    "marks_wrong": 0,
    "marks_unanswered": 0
  }
}
```

Only `active_questions` rows are processed. Remaining printed bubbles are ignored. Load a different JSON per exam with the same physical sheet.

---

## 10. Hardware Requirements

### Tier 1 — Starter (CPU-only inference)

Suitable for current PaperCreator scale. No upgrade needed.

| Spec | Value |
|---|---|
| vCPU | 4 cores |
| RAM | 8 GB |
| Storage | 50 GB NVMe SSD |
| GPU | Not required |
| Throughput | ~30–50 sheets/min |
| Estimated cost | $10–20/month |

### Tier 2 — Production (recommended upgrade when scaling)

| Spec | Value |
|---|---|
| vCPU | 8 cores (AMD EPYC or Intel Xeon with AVX-512) |
| RAM | 16 GB |
| Storage | 100 GB NVMe SSD |
| GPU | Not required |
| Throughput | ~150–200 sheets/min |
| Parallel workers | Celery + Redis |
| Handles | 1,000+ exams/day |
| Estimated cost | $48/month (Hetzner CCX33 or DigitalOcean CPU-Optimized) |

### Tier 3 — GPU (training only, not inference)

Do **not** buy a GPU server. Rent only when training models.

| Provider | GPU | Cost | Training time |
|---|---|---|---|
| Vultr | RTX 4090 | $1.50/hr | 2–5 hrs per model |
| Google Colab | T4 | Free tier | Works for small datasets |
| Lambda Cloud | A10 | $0.60/hr | Good mid-range option |

**Estimated one-time training cost:** $10–15 for both EfficientNet-B0 and YOLO11n combined.

> **Rule:** Train on rented GPU, deploy weights to your VPS, run inference on CPU forever after.

### RAM usage during inference

| Component | RAM used |
|---|---|
| OMRChecker (OpenCV) | ~200 MB |
| YOLO11n model loaded | ~150 MB |
| EfficientNet-B0 loaded | ~80 MB |
| FastAPI + workers | ~300 MB |
| **Total** | **~730 MB** |

Your current VPS with 8 GB RAM can comfortably run all four components simultaneously.

---

## 11. Training Plan

### Phase 1 — Data Collection (Week 1–2)

```
1. Print 50–100 exam sheets using PaperCreator's existing template
2. Have students fill them in normally (pen and pencil both)
3. Manually verify and record the correct answers
4. Run OMRChecker to extract bubble ROI crops
5. Sort crops into /training_data/filled/ and /training_data/empty/
```

Target: **500 filled + 500 empty** minimum per paper type.

### Phase 2 — YOLO11n Training (Week 2)

```
1. Annotate 200–300 full sheet images using Roboflow or LabelImg
   (draw bounding boxes around each bubble)
2. Export dataset in YOLO format
3. Train YOLO11n for 100 epochs on rented GPU
4. Validate mAP on held-out sheets
5. Export .pt weights → copy to VPS
```

### Phase 3 — EfficientNet-B0 Training (Week 2–3)

```
1. Use bubble crops from Phase 1
2. Train for 20 epochs (backbone frozen) + 10 epochs (fine-tuned)
3. Target: >96% validation accuracy
4. Save as .h5 → copy to VPS
```

### Phase 4 — Continuous Retraining Loop

Every Vision LLM resolution is automatically saved to `retraining_data/`. After accumulating 200–300 new LLM-resolved cases, retrain EfficientNet-B0 on the expanded dataset. Over time, the CNN learns your specific paper characteristics and the LLM fallback rate drops from ~3% toward ~0.5%.

---

## 12. Integration with PaperCreator

### FastAPI Microservice

```
papercreator-app     (Angular SSR)
papercreator-web-api (ASP.NET Core)  ──→  omr-service (FastAPI, Python)
ums-api              (ASP.NET Core)
identity-server
```

```python
# omr_service/main.py
from fastapi import FastAPI, UploadFile, File
import json

app = FastAPI()

@app.post("/process-sheet")
async def process_sheet(
    image: UploadFile = File(...),
    exam_config: str = Form(...)
):
    config = json.loads(exam_config)
    image_bytes = await image.read()

    # Run 4-stage pipeline
    preprocessed = preprocess_sheet_bytes(image_bytes)
    bubbles = detect_bubbles(preprocessed, config)
    classified = classify_all_bubbles(bubbles)
    final = postprocess_answers(classified, config)
    anomalies = detect_sheet_anomalies(final)

    return {
        "answers": final,
        "anomalies": anomalies,
        "flagged_count": sum(1 for a in final if a.get("flagged")),
        "score": calculate_score(final, config["answer_key"], config["scoring"])
    }
```

### Docker Compose addition

```yaml
omr-service:
  build: ./omr-service
  container_name: omr-service
  restart: unless-stopped
  volumes:
    - ./omr-models:/app/models
    - ./omr-training-data:/app/retraining_data
  environment:
    - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
  networks:
    - papercreator-net
```

### ASP.NET Core call

```csharp
public async Task<OmrResult> ProcessSheetAsync(IFormFile image, ExamConfig config)
{
    using var form = new MultipartFormDataContent();
    form.Add(new StreamContent(image.OpenReadStream()), "image", image.FileName);
    form.Add(new StringContent(JsonSerializer.Serialize(config)), "exam_config");

    var response = await _httpClient.PostAsync("http://omr-service:8000/process-sheet", form);
    var json = await response.Content.ReadAsStringAsync();
    return JsonSerializer.Deserialize<OmrResult>(json);
}
```

---

## 13. Accuracy Expectations

| Input type | Stage handling | Expected accuracy |
|---|---|---|
| 300 DPI scanner | OpenCV mostly | 98–99% |
| Phone photo, good light | OpenCV + CNN | 95–97% |
| Phone photo, poor light | CNN + LLM fallback | 92–95% |
| Xerox / photocopy | CNN + LLM fallback | 93–96% |

**Per-layer accuracy contribution:**

| Source | % of bubbles | Accuracy |
|---|---|---|
| OpenCV (clear marks) | ~85% | 97–98% |
| EfficientNet-B0 (ambiguous) | ~12% | 96–98% |
| Vision LLM (edge cases) | ~3% | 99%+ |
| **Combined system** | **100%** | **~98.5%** |

> After retraining on 1,000+ resolved cases, combined accuracy reaches 99%+.

---

## 14. Recommended Roadmap

### Phase 1 — Foundation (Month 1)

- [ ] Clone and run OMRChecker on sample PaperCreator sheets
- [ ] Design exam template with 4 corner registration markers
- [ ] Build the JSON template config structure
- [ ] Wrap OMRChecker in FastAPI microservice
- [ ] Integrate with ASP.NET Core via HTTP call

### Phase 2 — YOLO11n (Month 2)

- [ ] Collect 200–300 annotated sheet images
- [ ] Train YOLO11n on rented GPU (~$10–15 total)
- [ ] Replace OMRChecker's contour detection for phone input
- [ ] Validate on real exam sheets

### Phase 3 — EfficientNet-B0 (Month 2–3)

- [ ] Collect 1,000+ bubble crops from verified sheets
- [ ] Train EfficientNet-B0 (~$5–8 GPU cost)
- [ ] Add 3-layer confidence cascade to FastAPI service
- [ ] Add Claude Sonnet API fallback for low-confidence marks

### Phase 4 — Post-processing + Hardening (Month 3)

- [ ] Implement rule-based conflict resolver (multi-fill, erasure detection)
- [ ] Add DBSCAN clustering for free-form sheet support
- [ ] Add anomaly detection (all-same-answer, high flag rate)
- [ ] Build retraining data pipeline (auto-save LLM-resolved cases)

### Phase 5 — Production (Month 4+)

- [ ] Upgrade VPS to 8-core 16GB if throughput needed
- [ ] Add Celery + Redis for parallel sheet processing
- [ ] Add Qwen2.5-VL-7B offline Vision LLM (eliminates API cost)
- [ ] Continuous retraining loop every 200–300 new cases

---

## Quick Reference

```
Models to train:   YOLO11n + EfficientNet-B0
Training cost:     ~$10–15 one-time (rented GPU)
Inference:         CPU-only VPS, no GPU needed
RAM required:      ~730 MB total for all components
Monthly VPS cost:  $10–20 (starter) / $48 (production)
Throughput:        30–50 sheets/min (starter) / 150–200 (production)
Accuracy target:   98–99% scanner, 95–97% phone photo
Framework:         FastAPI (Python) → ASP.NET Core HTTP call
License:           OMRChecker MIT, YOLO11 AGPL-3.0, EfficientNet Apache 2.0
```

---

*Document generated for PaperCreator OMR feature planning — August 2026*
