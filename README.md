# ISTVT-Inspired Deepfake Detection

## Overview

A CNN-Transformer based deepfake detection pipeline inspired by ISTVT.

## Pipeline

Video
→ Frame Sampling
→ Face Detection
→ ResNet50 Feature Extraction
→ Transformer Temporal Modeling
→ Deepfake Classification
→ GradCAM Explanation


## Results

Accuracy: 90%
AUC: 0.945


## Features

- Video-level deepfake detection
- Temporal modeling
- GradCAM interpretability
- Frame importance scoring