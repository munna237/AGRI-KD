# AGRI-KD: A Lightweight Multi-Task Framework for Rice Disease Diagnosis

[![Paper](https://img.shields.io/badge/Paper-Smart%20Ag%20Tech-green)](https://doi.org/PENDING)
[![Dataset](https://img.shields.io/badge/Dataset-Mendeley%20Data-blue)](https://doi.org/10.17632/86s4jzj2m4.1)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

Official implementation of **"AGRI-KD: A Lightweight Multi-Task Framework for Rice Disease Diagnosis with Localization and Severity Quantification"**, submitted to *Smart Agricultural Technology* (Elsevier).

## Overview

AGRI-KD is a multi-task knowledge distillation framework that compresses an ensemble of three teacher CNNs into a single lightweight MobileNetV3-Small student (1.73M parameters, 6.8 MB) capable of producing **five simultaneous outputs** from a single forward pass:

| Output | Metric | Value |
|--------|--------|-------|
| Disease Classification | Accuracy | 91.9% |
| Spatial Localization | Dice Coefficient | 0.574 |
| Infected Area % (IAP) | MAE | 1.75% |
| Severity Grading (5 levels) | Accuracy | 90.0% |
| Lesion Counting | R² | 0.851 |

The model runs at **18.9 ms** on a budget smartphone (MediaTek Helio 55), enabling real-time in-field diagnosis for smallholder farmers.

<p align="center">
  <img src="figures/fig_architecture.png" width="90%" alt="AGRI-KD Architecture">
</p>

## Key Results

### Ablation Study

| Config | KD | Att | Agri | Acc (%) | Dice | IAP MAE | Sev Acc |
|--------|----|-----|------|---------|------|---------|---------|
| A1: Baseline | | | | 92.29 | 0.258 | 50.80% | 18.6% |
| A2: +Agri | | | ✓ | 91.20 | 0.243 | 1.77% | 90.0% |
| A3: +Attention | | ✓ | | 92.22 | 0.545 | 50.85% | 13.3% |
| A4: +KD | ✓ | | | 92.50 | 0.384 | 53.34% | 4.9% |
| A5: +KD+Att | ✓ | ✓ | | 92.70 | 0.554 | 53.27% | 5.1% |
| **A6: Full AGRI-KD** | ✓ | ✓ | ✓ | **91.88** | **0.574** | **1.75%** | **90.0%** |

### Deployment Benchmarks

| Platform | Latency | FPS | RAM |
|----------|---------|-----|-----|
| NVIDIA L4 (GPU) | 6.4 ms | 157 | — |
| Intel Xeon (CPU) | 12.1 ms | 82 | — |
| Helio 55 Smartphone (1-thread) | 18.9 ms | 53 | 20.5 MB |
| Helio 55 Smartphone (4-thread) | 10.8 ms | 92 | 20.3 MB |

### Attention Visualization

<p align="center">
  <img src="figures/fig_attention_viz.png" width="90%" alt="Attention Comparison">
</p>

## Repository Structure

```
AGRI-KD/
├── README.md
├── LICENSE
├── requirements.txt
│
├── notebooks/
│   └── AGRIKD_Q1_Complete.ipynb      # Full training + evaluation pipeline
│
├── models/
│   ├── weights/
│   │   ├── A1_Baseline.pth           # Baseline MobileNetV3-Small
│   │   ├── A2_Agricultural.pth       # +Agricultural heads
│   │   ├── A3_Attention.pth          # +Attention supervision
│   │   ├── A4_KD.pth                 # +Knowledge distillation
│   │   ├── A5_KD_Attention.pth       # +KD+Attention
│   │   └── A6_Full_AGRIKD.pth        # Full AGRI-KD (recommended)
│   └── export/
│       ├── agrikd.tflite             # TFLite for Android deployment
│       └── agrikd_mobile.onnx        # ONNX for cross-platform inference
│
├── results/
│   ├── ablation_results.csv          # Main ablation study (Table 3)
│   ├── deployment_benchmarks.csv     # Efficiency metrics (Table 6)
│   ├── smartphone_benchmark.txt      # Helio 55 TFLite benchmark output
│   ├── sota_comparison/
│   │   └── sota_comparison.csv       # KD method comparison (Table 4)
│   ├── attention_comparison/
│   │   └── attention_comparison.csv  # Attention method comparison (Table 5)
│   ├── significance_analysis/
│   │   ├── cv_results.csv            # 5-fold cross-validation
│   │   ├── multi_seed_results.csv    # Multi-seed stability
│   │   ├── per_class_mcnemar.csv     # Per-class McNemar's test
│   │   └── hp_tuning_results.csv     # Hyperparameter sensitivity
│   └── yolo_detection/
│       └── results.csv               # YOLOv8-Nano comparison
│
├── figures/
│   ├── fig_architecture.png          # Framework diagram
│   ├── fig_attention_viz.png         # Attention visualization
│   ├── fig_attention_bars.png        # Attention quantitative comparison
│   ├── fig_severity_cm.png           # Severity confusion matrix
│   └── fig_dataset_samples.png       # Dataset examples
│
└── benchmark/
    ├── benchmark_phone.py            # Termux/ONNX phone benchmark script
    └── benchmark_tflite.sh           # TFLite benchmark tool commands
```

## Quick Start

### Requirements

```bash
pip install torch torchvision timm albumentations scikit-learn matplotlib seaborn
```

### Inference (Single Image)

```python
import torch
import timm
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Load model
class StudentModel(torch.nn.Module):
    def __init__(self, num_classes=6, num_severity_grades=5):
        super().__init__()
        self.backbone = timm.create_model('mobilenetv3_small_100', pretrained=False)
        self.feature_dim = self.backbone.classifier.in_features
        self.backbone.classifier = torch.nn.Identity()
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.3), torch.nn.Linear(self.feature_dim, num_classes))
        self.iap_head = torch.nn.Sequential(
            torch.nn.Dropout(0.2), torch.nn.Linear(self.feature_dim, 64),
            torch.nn.ReLU(), torch.nn.Linear(64, 1), torch.nn.Sigmoid())
        self.severity_head = torch.nn.Sequential(
            torch.nn.Dropout(0.2), torch.nn.Linear(self.feature_dim, 32),
            torch.nn.ReLU(), torch.nn.Linear(32, num_severity_grades))
        self.count_head = torch.nn.Sequential(
            torch.nn.Dropout(0.2), torch.nn.Linear(self.feature_dim, 32),
            torch.nn.ReLU(), torch.nn.Linear(32, 1), torch.nn.ReLU())
        self.spatial_features = None
        self.backbone.blocks.register_forward_hook(
            lambda m, inp, out: setattr(self, 'spatial_features', out))
        self.attention_conv = torch.nn.Sequential(
            torch.nn.Conv2d(576, 128, 1, bias=False), torch.nn.BatchNorm2d(128), torch.nn.ReLU(True),
            torch.nn.Conv2d(128, 32, 1, bias=False), torch.nn.BatchNorm2d(32), torch.nn.ReLU(True),
            torch.nn.Conv2d(32, 1, 1), torch.nn.Sigmoid())

    def forward(self, x):
        features = self.backbone(x)
        attention = self.attention_conv(self.spatial_features).squeeze(1)
        return {
            'cls': self.classifier(features),
            'iap': self.iap_head(features).squeeze(-1) * 100,
            'severity': self.severity_head(features),
            'count': self.count_head(features).squeeze(-1),
            'attention': attention
        }

# Load weights
model = StudentModel()
model.load_state_dict(torch.load('models/weights/A6_Full_AGRIKD.pth', map_location='cpu'))
model.eval()

# Prepare image
CLASS_NAMES = ['Healthy', 'Blast', 'Brown spot', 'Leaf smut', 'Rice Tungro', 'Sheath blight']
SEVERITY_NAMES = ['Minimal (<5%)', 'Mild (5-15%)', 'Moderate (15-30%)', 'Severe (30-50%)', 'Critical (>50%)']

transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

img = cv2.imread('your_image.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
tensor = transform(image=img)['image'].unsqueeze(0)

# Inference
with torch.no_grad():
    out = model(tensor)

disease = CLASS_NAMES[out['cls'].argmax(1).item()]
iap = out['iap'].item()
severity = SEVERITY_NAMES[out['severity'].argmax(1).item()]
count = out['count'].item()

print(f"Disease:  {disease}")
print(f"IAP:      {iap:.1f}%")
print(f"Severity: {severity}")
print(f"Lesions:  {count:.0f}")
```

### Training (Full Pipeline)

Open `notebooks/AGRIKD_Q1_Complete.ipynb` in Google Colab and follow the cells sequentially. The notebook handles:
1. Dataset loading and preprocessing
2. Teacher fine-tuning (ResNet-50, DenseNet-121, EfficientNet-B0)
3. Knowledge extraction (soft labels, attention maps)
4. Student training with composite loss
5. Ablation study (A1–A6 configurations)
6. Evaluation and statistical analysis

### Smartphone Deployment

```bash
# Export to TFLite (in Colab)
# See notebook Section: "Model Export"

# Benchmark on Android via Termux
pkg install wget
wget https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_aarch64_benchmark_model -O benchmark_model
chmod +x benchmark_model
./benchmark_model --graph=models/export/agrikd.tflite --num_threads=1 --num_runs=200
```

## Dataset

The **RiceLeafDiseaseBD** dataset is publicly available:

- **Source**: [Mendeley Data (DOI: 10.17632/86s4jzj2m4.1)](https://doi.org/10.17632/86s4jzj2m4.1)
- **Size**: 9,769 images, 74,836 bounding box annotations
- **Classes**: Blast, Brown Spot, Healthy, Leaf Smut, Rice Tungro, Sheath Blight
- **Format**: YOLO annotation format
- **Collection**: Field-collected in Gazipur, Bangladesh

## Citation

If you use this code or dataset, please cite:

```bibtex
@article{munna2026agrikd,
  title={AGRI-KD: A Lightweight Multi-Task Framework for Rice Disease Diagnosis 
         with Localization and Severity Quantification},
  author={Munna, Mohammad Mehedi Hasan and Tulona, Reefka Fabliha and Kabir, Acramul Haque},
  journal={Smart Agricultural Technology},
  year={2026},
  publisher={Elsevier}
}

@misc{kabir2024riceleafdiseasebd,
  title={RiceLeafDiseaseBD Dataset},
  author={Kabir, Acramul Haque},
  year={2024},
  howpublished={\url{https://doi.org/10.17632/86s4jzj2m4.1}},
  note={Mendeley Data}
}
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgments

- Google Colaboratory for computational resources
- [timm](https://github.com/huggingface/pytorch-image-models) for backbone architectures
- [Albumentations](https://albumentations.ai/) for data augmentation
- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv8 comparison
