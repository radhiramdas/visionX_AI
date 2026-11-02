# VisionX AI — AI-Powered Multi-Disease Eye Screening & Clinical Decision Support

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Flask-000000.svg)](https://flask.palletsprojects.com/)
[![Deep Learning](https://img.shields.io/badge/Deep%20Learning-PyTorch%20%7C%20timm-ee4c2c.svg)](https://pytorch.org/)
[![Model](https://img.shields.io/badge/Architecture-EfficientNet--B0-green.svg)](https://arxiv.org/abs/1905.11946)
[![Explainability](https://img.shields.io/badge/XAI-Grad--CAM-yellow.svg)](https://arxiv.org/abs/1610.02391)
[![Database](https://img.shields.io/badge/Database-MySQL-00758f.svg)](https://www.mysql.com/)

**VisionX AI** is an end-to-end clinical ophthalmological decision support system designed for early multi-disease screening from retinal fundus photographs. Powered by a fine-tuned **EfficientNet-B0** deep convolutional network and **Grad-CAM (Gradient-Weighted Class Activation Mapping)**, the platform empowers hospital intake staff and consulting ophthalmologists to screen, track, and document retinal conditions with high accuracy and full visual interpretability.

---

## 🌟 Key Capabilities

* **7-Class Multi-Disease Detection**: Accurately classifies retinal fundus images across seven clinical conditions.
* **Explainable AI (Grad-CAM Heatmaps)**: Visualizes the exact morphological regions (hemorrhages, optic cup cupping, drusen, exudates) that drive the AI diagnosis.
* **Intelligent Intake Gatekeeper**: Pre-inference optical and semantic validation filter that automatically detects and rejects non-retinal images (diagrams, portraits, pets, vehicles, or blurred photos) before model inference.
* **Role-Based Access Control (RBAC)**: Secure multi-tier authentication with dedicated portals for:
  * **Hospital Staff**: Patient registration, longitudinal search, screening intake, and exam creation.
  * **Doctor / Clinician**: Longitudinal exam review, disease progression tracking, and clinical sign-off.
  * **Administrator**: System statistics, hospital analytics, and user account management.
* **Longitudinal Patient Tracking**: Preserves unique patient IDs across multiple visits, allowing clinicians to compare new examinations with historical records to monitor disease progression or treatment response.
* **Automated Clinical PDF Reports**: Generates downloadable, standardized clinical PDF reports with patient demographics, input scans, Grad-CAM overlays, AI confidence metrics, and doctor review fields.
* **Modern Clinical UI**: Clean, responsive ophthalmological dashboard with light/dark theme support.

---

## 👁️ Supported Retinal Conditions

| Condition | Clinical Manifestations & Biomarkers Identified |
| :--- | :--- |
| **Diabetic Retinopathy (DR)** | Microaneurysms, intraretinal blot hemorrhages, hard/soft lipid exudates, neovascularization. |
| **Glaucoma** | Enlarged optic cup-to-disc ratio (CDR > 0.6), neuroretinal rim thinning, optic disc pallor. |
| **Cataract** | Optical cloudiness, diffuse blurriness of retinal vasculature, yellowish media discoloration. |
| **Age-Related Macular Degeneration (AMD)** | Central macular drusen deposits, geographic atrophy, pigmentary changes. |
| **Hypertensive Retinopathy** | Arteriolar narrowing (silver/copper wiring), arteriovenous (AV) crossing changes, flame hemorrhages. |
| **Pathological Myopia** | Chorioretinal atrophy, tilted optic disc, posterior staphyloma, lacquer cracks. |
| **Normal Fundus** | Intact neuroretinal rim, well-defined macula and foveal reflex, normal vascular caliber. |

---

## 📊 Model Evaluation & Benchmarks

The core diagnostic classifier (**EfficientNet-B0**) was trained and evaluated on a diverse benchmark dataset of fundus photography:

* **Evaluation Split**: 1,236 test images
* **Overall Test Accuracy**: **86.08%**
* **Macro ROC-AUC**: **97.63%**
* **Weighted F1-Score**: **85.70%**

### Per-Class Performance Breakdown

| Retinal Class | Test Support | Precision | Recall | F1-Score | ROC-AUC (OvR) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Diabetic Retinopathy (DR)** | 516 | **98.66%** | **99.81%** | **99.23%** | **99.99%** |
| **Cataract** | 112 | **83.20%** | **92.86%** | **87.76%** | **99.17%** |
| **Pathological Myopia** | 102 | **92.05%** | **79.41%** | **85.26%** | **99.25%** |
| **Normal Fundus** | 179 | **76.39%** | **92.18%** | **83.54%** | **98.62%** |
| **Hypertensive Retinopathy** | 94 | **65.00%** | **69.15%** | **67.01%** | **96.97%** |
| **Glaucoma** | 156 | **75.00%** | **57.69%** | **65.22%** | **95.17%** |
| **AMD** | 77 | **67.69%** | **57.14%** | **61.97%** | **94.21%** |

---

## 🏗️ System Architecture & Workflow

```
                                [ Staff / Doctor Portal ]
                                            │
                                  (Upload Fundus Scan)
                                            ▼
                           ┌───────────────────────────────────┐
                           │   Pre-Screening Gatekeeper        │
                           │   - Aspect ratio & resolution     │
                           │   - Canvas brightness & contrast  │
                           │   - Semantic domain verification  │
                           └─────────────────┬─────────────────┘
                                             │ (Valid Fundus)
                                             ▼
                           ┌───────────────────────────────────┐
                           │   EfficientNet-B0 Inference       │
                           │   - Letterbox Resize (224x224)    │
                           │   - 7-Class Softmax Probabilities │
                           └─────────────────┬─────────────────┘
                                             │
                      ┌──────────────────────┴──────────────────────┐
                      ▼                                             ▼
         ┌─────────────────────────┐                   ┌─────────────────────────┐
         │     Grad-CAM Engine     │                   │     MySQL Database      │
         │  - Conv Head Gradients  │                   │  - Patients & History   │
         │  - Heatmap Generation   │                   │  - Exam & Prediction ID │
         └────────────┬────────────┘                   └────────────┬────────────┘
                      │                                             │
                      └──────────────────────┬──────────────────────┘
                                             ▼
                           ┌───────────────────────────────────┐
                           │   Evaluation & PDF Report         │
                           │   - Interactive Result Dashboard  │
                           │   - Standardized PDF Clinical Doc │
                           └───────────────────────────────────┘
```

---

## 🛠️ Technology Stack

* **Deep Learning & Computer Vision**: PyTorch, timm, Torchvision, OpenCV, NumPy, Pillow
* **Explainability (XAI)**: Grad-CAM (Targeting final EfficientNet-B0 convolutional stage)
* **Backend**: Python 3.10+, Flask, Werkzeug, ReportLab (PDF Engine)
* **Database**: MySQL Server, `mysql-connector-python`
* **Frontend**: Vanilla JavaScript (ES6+), HTML5, Responsive CSS3 (Custom Design System with Light/Dark Theme)
* **Security**: `werkzeug.security` (PBKDF2-SHA256 password hashing), session-based RBAC, SQL injection protection

---

## 📁 Repository Structure

```
VisionX_AI/
├── backend/
│   ├── app.py                   # Flask server, route controllers & API endpoints
│   ├── prediction.py            # Preprocessing, pre-screening gatekeeper & inference
│   ├── model_loader.py          # PyTorch checkpoint loader with fallback resolution
│   ├── gradcam.py               # Grad-CAM heatmap generation & image blending
│   ├── db_operations.py         # MySQL connection pooling & CRUD operations
│   ├── report_generator.py      # Standardized clinical PDF report generator
│   ├── generated/               # Output directory for Grad-CAM heatmaps
│   ├── reports/                 # Output directory for clinical PDF reports
│   └── uploads/                 # Storage for uploaded patient retinal scans
├── frontend/
│   ├── static/
│   │   ├── css/style.css        # Clinical UI design system & responsive layout
│   │   └── js/                  # Authentication, upload, and dashboard scripts
│   └── templates/
│       ├── index.html           # Multi-role authentication portal (Staff/Doctor/Admin)
│       ├── staff_dashboard.html # Patient registration, lookup & screening workflow
│       ├── doctor_dashboard.html# Clinical review & longitudinal patient tracking
│       ├── admin_dashboard.html # System metrics & hospital administration
│       ├── upload.html          # Retinal image upload with live pre-check
│       ├── result.html          # Diagnostic evaluation summary & Grad-CAM viewer
│       ├── 403.html / 404.html  # Security & navigation error views
├── evaluation_results/
│   └── metrics.json             # Benchmark evaluation metrics on 1,236 test scans
├── tuning_results/              # Checkpoint storage for trained models
│   └── experiment_3_finetuned/
│       └── best_model.pth       # Deployed EfficientNet-B0 model weights
├── dataset.py                   # PyTorch dataset loader with letterbox transforms
├── model.py                     # EfficientNet-B0 architecture definition
├── train.py                     # Model training & fine-tuning script
├── evaluate.py                  # Test set evaluation and metrics computation
└── README.md                    # Project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites
* Python 3.10 or higher
* MySQL Server (8.0+ recommended)
* Git

### 2. Clone the Repository
```bash
git clone https://github.com/radhiramdas/visionX_AI.git
cd visionX_AI
```

### 3. Set Up Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install torch torchvision timm flask mysql-connector-python pillow numpy opencv-python reportlab matplotlib scikit-learn
```

### 5. Configure MySQL Database
Create your database and update credentials in `.env`:
```env
DB_HOST=localhost
DB_USER=root
DB_PASSWORD=your_password
DB_NAME=visionx_db
DB_PORT=3306
SECRET_KEY=your_secure_random_flask_secret
```

The database tables (`users`, `patients`, `examinations`, `eye_images`, `ai_predictions`) are automatically verified and initialized on application launch.

### 6. Run the Application
```bash
python backend/app.py
```
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 👥 User Roles & Clinical Workflow

1. **Staff Portal (`/staff/dashboard`)**:
   * Register a new patient with demographics (Name, Age, Location, Contact).
   * Or search for an existing patient by Name or Patient ID.
   * View the patient's previous screening summary before initiating a new exam.
   * Upload fundus scan $\rightarrow$ System performs automated quality and semantic validation $\rightarrow$ Runs EfficientNet-B0 inference.
2. **Consulting Doctor Portal (`/doctor/dashboard`)**:
   * View assigned patient list and recent examination logs.
   * Compare previous vs. latest examinations side-by-side to track disease progression.
   * Inspect high-resolution Grad-CAM heatmaps.
   * Add clinical consultation notes and download official PDF reports.
3. **Admin Portal (`/admin/dashboard`)**:
   * View high-level hospital screening analytics (total patients, exams, disease distributions).
   * Manage user accounts and system configuration.

---

## 📄 License & Disclaimer

**Clinical Disclaimer**: VisionX AI is intended as an investigational medical decision-support and screening tool. All diagnostic evaluations and treatment decisions must be confirmed by a licensed ophthalmologist or qualified healthcare provider.
