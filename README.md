# AGRI-KD: A Lightweight Multi-Task Framework for Rice Disease Diagnosis

[![Paper](https://img.shields.io/badge/Paper-Smart%20Ag%20Tech-green)](https://doi.org/PENDING)
[![Dataset](https://img.shields.io/badge/Dataset-Mendeley%20Data-blue)](https://doi.org/10.17632/86s4jzj2m4.2)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

Official implementation of **"AGRI-KD: A Lightweight Multi-Task Framework for Rice Disease Diagnosis with Localization and Severity Quantification"**, submitted to *Smart Agricultural Technology* (Elsevier).

## Overview

AGRI-KD compresses an ensemble of three teacher CNNs into a single lightweight MobileNetV3-Small student (1.73M parameters, 6.97 MB) producing **five simultaneous outputs** from one forward pass:

| Output | Metric | Value |
|--------|--------|-------|
| Disease Classification | Accuracy | 88.97% |
| Spatial Localization | Dice Coefficient | 0.563 |
| Infected Area % (leaf-area IAP) | MAE / R² | 8.28% / 0.705 |
| Severity Grading (5 levels) | Accuracy / F1-macro | 67.2% / 0.606 |
| Lesion Counting | MAE / R² | 0.81 / 0.574 |

The model runs at **14.8 ms** on a budget smartphone (Helio 55, MediaTek Helio G92 SoC), enabling real-time in-field diagnosis for smallholder farmers.

Infected area percentage is normalised by **leaf area**, not image area, so the measure does not vary with shooting distance and the severity thresholds correspond to published agronomic scales.

<p align="center">
  <img src="figures/fig_architecture.png" width="90%" alt="AGRI-KD Architecture">
</p>

## Key Results

### Ablation Study

| Config | KD | Att | Agri | Acc (%) | Dice | Pointing (%) | IAP MAE | Sev Acc |
|--------|----|-----|------|---------|------|--------------|---------|---------|
| A1: Baseline | | | | 88.70 | 0.299 | 20.1 | — | — |
| A2: +Agri | | | ✓ | 89.18 | 0.289 | 16.0 | **7.86%** | **68.2%** |
| A3: +Attention | | ✓ | | **89.73** | **0.568** | **78.4** | — | — |
| A4: +KD | ✓ | | | 88.56 | 0.346 | 40.4 | — | — |
| A5: +KD+Att | ✓ | ✓ | | 88.56 | 0.565 | 77.3 | — | — |
| **A6: Full AGRI-KD** | ✓ | ✓ | ✓ | 88.97 | 0.563 | 78.3 | 8.28% | 67.2% |

Attention supervision drives localisation (A3 vs A1: Dice 0.299 → 0.568, Cohen's *d* = 1.59). Agricultural heads drive quantification. Configurations without the corresponding head produce no output for that column.

### Attention Localization

| Method | IoU | Dice | Pointing (%) |
|--------|-----|------|--------------|
| Baseline Attention | 0.187 | 0.295 | 18.3 |
| Score-CAM | 0.142 | 0.202 | 30.7 |
| Grad-CAM | 0.307 | 0.429 | 66.0 |
| Grad-CAM++ | 0.386 | 0.524 | 77.3 |
| Layer-CAM | 0.385 | 0.522 | 77.3 |
| **AGRI-KD (learned)** | **0.419** | **0.564** | **78.7** |

Evaluated on 300 diseased test samples at the native 7×7 resolution. AGRI-KD requires no gradient computation at inference, unlike every post-hoc method above.

### Deployment Benchmarks

Helio 55 (MediaTek Helio G92, Android 15), TensorFlow Lite FP32, `benchmark_model`, 20 warm-up + 200 measured runs, three cold-start repetitions.

| Configuration | Median | Range | CV | FPS | Memory |
|---------------|--------|-------|-----|-----|--------|
| XNNPACK, 1 thread | **14.80 ms** | 14.73–14.85 | 1.0% | 68 | 20.8 MB |
| XNNPACK, 2 threads | 8.90 ms | 8.87–8.96 | 20.8% | 112 | 20.9 MB |
| XNNPACK, 4 threads | 7.97 ms | 7.67–8.67 | 15.5% | 125 | 20.3 MB |
| No delegate, 1 thread | 20.88 ms | — | 1.0% | 48 | 14.0 MB |

The XNNPACK delegate provides a **1.41× speedup** at a cost of ~6.8 MB memory. Multi-threaded configurations are faster on average but substantially less stable, reflecting the heterogeneous big.LITTLE core layout; we report single-thread as the primary figure.

### Attention Visualization

<p align="center">
  <img src="figures/fig_attention_viz.png" width="90%" alt="Attention Comparison">
</p>

## Repository Structure

```
agri-kd/
├── README.md
├── LICENSE
├── requirements.txt
├── setup.py
│
├── agrikd/                          # importable package
│   ├── models/                      # student, teachers, ensemble, leaf segmenter
│   ├── losses/                      # attention, KD, contrastive, composite objective
│   ├── data/                        # dataset, split handling, target construction
│   ├── engine/                      # training loops, per-sample evaluation
│   ├── utils/                       # seeding, checkpointing, metrics
│   └── config.py                    # configuration and the six ablation settings
│
├── scripts/
│   ├── 00_train_leafseg.py          # leaf segmenter for the IAP denominator
│   ├── 01_build_targets.py          # leaf masks → leaf-area IAP targets
│   ├── 02_train.py                  # teachers + six ablation configurations
│   ├── 03_evaluate.py               # per-sample test predictions and metrics
│   └── 04_export_tflite.py          # TFLite export with numerical verification
│
├── configs/
│   └── default.yaml                 # settings reported in the paper
│
├── annotations/                     # 405 hand-annotated leaf polygons (CC BY 4.0)
│   ├── manifest.csv
│   ├── images/
│   └── masks/
│
├── figures/                         # publication figures
│
└── docs/
    ├── DATASET.md                   # data preparation and layout
    └── REPRODUCIBILITY.md           # expected run-to-run variation
```

## Quick Start

### Install

```bash
git clone https://github.com/<org>/agri-kd.git
cd agri-kd
pip install -e .
```

Python ≥ 3.9, PyTorch ≥ 2.0. Training needs a CUDA GPU; inference runs on CPU.

### Inference

```python
import cv2, torch
from agrikd import StudentModel
from agrikd.data import eval_transform

CLASS_NAMES = ['Healthy', 'Blast', 'Brown spot', 'Leaf smut', 'Rice Tungro', 'Sheath blight']
SEVERITY = ['Minimal (<5%)', 'Mild (5-15%)', 'Moderate (15-30%)',
            'Severe (30-50%)', 'Critical (>50%)']

model = StudentModel(pretrained=False)
model.load_state_dict(torch.load('models/weights/A6_Full_AGRIKD.pth', map_location='cpu'))

image = cv2.cvtColor(cv2.imread('your_image.jpg'), cv2.COLOR_BGR2RGB)
tensor = eval_transform()(image=image)['image'].unsqueeze(0)

out = model.predict(tensor)
print(f"Disease:  {CLASS_NAMES[out['class_id'].item()]}")
print(f"IAP:      {out['iap_percent'].item():.1f}% of leaf area")
print(f"Severity: {SEVERITY[out['severity_grade'].item()]}")
print(f"Lesions:  {out['lesion_count'].item():.1f}")
```

### Full pipeline

```bash
python scripts/00_train_leafseg.py --config configs/default.yaml
python scripts/01_build_targets.py --config configs/default.yaml
python scripts/02_train.py         --config configs/default.yaml
python scripts/03_evaluate.py      --config configs/default.yaml
python scripts/04_export_tflite.py --weights outputs/checkpoints/A6_Full_AGRIKD_best.pth
```

Training and evaluation checkpoint every epoch and resume automatically after an interruption. Train a single configuration with `--only A3_AttentionOnly`.

### Smartphone deployment

```bash
# on Android via Termux
pkg install wget
wget https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_aarch64_benchmark_model -O benchmark_model
chmod +x benchmark_model

./benchmark_model --graph=agrikd_5out.tflite --num_threads=1 \
  --num_runs=200 --warmup_runs=20 --use_xnnpack=true \
  --report_peak_memory_footprint=true
```

## Reproducing our numbers

**Your results will differ slightly from the paper.** Two independent runs of this code, same seed and same hardware, differed by up to 1.7 percentage points on severity accuracy and 0.5 on classification accuracy.

Treat **±1 pp on accuracy metrics** and **±0.01 on Dice and IoU** as successful reproduction. `docs/REPRODUCIBILITY.md` gives the measured variation, its sources, and which findings reproduce robustly.

To match our partition exactly, use the `split.csv` released with the paper rather than regenerating it — directory listing order is filesystem dependent, so a random seed alone does not reproduce split membership.

## Dataset

**RiceLeafDiseaseBD** is distributed separately and is not included here.

- **Source**: [Mendeley Data](https://doi.org/10.17632/86s4jzj2m4.2)
- **Size**: 9,769 images, 21,460 bounding box annotations
- **Classes**: Blast, Brown Spot, Healthy, Leaf Smut, Rice Tungro, Sheath Blight
- **Format**: YOLO annotation format
- **Collection**: Field-collected in Gazipur, Bangladesh

Use the current dataset version; see `docs/DATASET.md` for the expected layout.

Leaf-area IAP additionally requires a leaf mask per image, which the dataset does not provide. This repository ships 405 hand-annotated leaf polygons in `annotations/`, drawn on training-split images only, from which `scripts/00_train_leafseg.py` trains the segmenter used to generate the remaining masks.

## Citation

```bibtex
@article{munna2026agrikd,
  title={AGRI-KD: A Lightweight Multi-Task Framework for Rice Disease Diagnosis
         with Localization and Severity Quantification},
  author={Munna, Mohammad Mehedi Hasan and Tulona, Reefka Fabliha and
          Kabir, Acramul Haque},
  journal={Smart Agricultural Technology},
  year={2026},
  publisher={Elsevier}
}

@misc{kabir2026riceleafdiseasebd,
  title={RiceLeafDiseaseBD Dataset},
  author={Kabir, Acramul Haque},
  year={2026},
  howpublished={\url{https://doi.org/10.17632/86s4jzj2m4.2}},
  note={Mendeley Data}
}
```

## License

Code is licensed under the MIT License — see [LICENSE](LICENSE). The leaf polygon annotations in `annotations/` are released under CC BY 4.0.

## Acknowledgments

- Google Colaboratory for computational resources
- [timm](https://github.com/huggingface/pytorch-image-models) for backbone architectures
- [Albumentations](https://albumentations.ai/) for data augmentation
- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv8 comparison
