# XMal: Explainable Multimodal Android Malware Classification

XMal is a research framework for Android malware family classification that integrates behavioral features and image-based deep representations using multimodal feature fusion and machine learning.

## Key Components

- **Hybrid Feature Selection:** Mutual Information, Random Forest feature importance, and permutation-based selection.
- **Image Feature Extraction:** Deep feature extraction using ResNet50.
- **Multimodal Fusion:** Integration of behavioral features and image embeddings.
- **Malware Classification:** Random Forest-based classification of Android malware families.
- **Explainable AI:** Grad-CAM, Integrated Gradients, and Saliency Maps for model interpretation.
- **Behavioral Analysis:** Malware family-to-behavior mapping, t-SNE visualization, and K-Means clustering.

## Methodology

The framework follows the workflow:

1. Behavioral feature preprocessing and selection.
2. ResNet50-based image embedding extraction.
3. Multimodal feature fusion.
4. Hybrid feature selection on fused representations.
5. Random Forest-based malware classification.
6. Explainability and behavioral analysis.

## Repository Structure

```text
XMal/
├── embedding_extractor.py
├── retain_features.py
├── run_batch_inference.py
├── mapping.py
├── Grad_CAM visualization/
├── Integrated Gradients/
├── Saliency visualizations/
├── results/
├── predictions.csv
├── retain_features.csv
└── proposed.drawio.png
```

## Dataset

The framework is developed using the **CCCS-CIC-AndMal2020 Android malware dataset**. Behavioral features and corresponding malware visualization images are used for multimodal analysis.

Dataset files are not included in this repository.

## Technologies

Python, PyTorch, TensorFlow, Scikit-learn, ResNet50, NumPy, Pandas, Matplotlib, and Seaborn.

## Research Purpose

This repository provides the implementation and experimental resources associated with research on explainable multimodal Android malware classification.

## Author

**Ashwini Chikenakoppa**

GitHub: [Ashwini8765](https://github.com/Ashwini8765)

## Disclaimer

This repository is intended for academic research, defensive cybersecurity, and educational purposes.
