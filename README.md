# UMORSS: Uncertainty-aware Multimodal Ovarian Risk Scoring System

This repository contains the code implementation for UMORSS (Uncertainty-aware Multimodal Ovarian Risk Scoring System), a novel multimodal AI framework for automated ovarian cancer risk assessment that integrates uncertainty quantification. UMORSS combines deep learning-based ultrasound image analysis with clinical biomarkers to provide reliable prediction of malignancy risk while accounting for model uncertainty.

## Environment Setup

Install requirements:

```bash
pip install -r requirements.txt
```

## Project Structure

- `models/` - Model architecture implementations
  - `van.py` and `van2.py` - Vision Attention Network (VAN) model
- `single_test.py` - Testing script for single case prediction
- `ai-assistance.py` - Helper functions for combining doctor and AI predictions
- `LASSO.csv` - Feature coefficients from LASSO regression

## Usage

1. Single case testing:

```python
python single_test.py
```

This script demonstrates prediction on a single test image with:

- Phase 1 initial risk assessment using VAN model
- Phase 2 detailed analysis combining ultrasound imaging and clinical features
- Uncertainty estimation

2. AI-Doctor combined prediction:

```python
python ai-assistance.py
```

Provides functions to combine O-RADS scores from:

- Doctor's assessment
- AI model predictions
- Uncertainty measurements

## Model Details

The system uses a two-phase approach:

1. Initial screening using VAN for binary risk classification
2. Detailed analysis combining:
   - Deep learning features from ultrasound image
   - Clinical biomarkers and patient data
   - LASSO regression for feature selection
