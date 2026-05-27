# WDBC Classification Project

## Overview

This project applies multiple machine learning classification algorithms to classify breast cancer tumors as either **malignant** or **benign** using the Wisconsin Diagnostic Breast Cancer (WDBC) dataset from scikit-learn.

The project demonstrates:
- Binary classification
- Model evaluation
- Confusion matrix visualization
- Scatter plot visualization
- Comparison of multiple machine learning classifiers

---

## Dataset used

This project uses the built-in Breast Cancer dataset provided by scikit-learn.

Dataset details:
- 569 samples
- 30 numerical features
- 2 target classes:
  - Malignant
  - Benign

---

## Machine Learning Models Tested

The following classifiers were trained and evaluated:

- Support Vector Machine (SVM)
- Decision Tree
- K-Nearest Neighbors (KNN)
- Random Forest

---

## Evaluation Metrics

Each model was evaluated using:

- Accuracy
- Precision
- Recall

---

## Project Structure

```text
wdbc_classification_project/
│
├── wdbc_classification.py
├── README.md
├── requirements.txt
├── short_report.txt
├── LICENSE
├── wdbc_classification_matrix.png
└── wdbc_classification_scatter.png
```

---

## Requirements Installation

Install the required Python libraries:

```bash
pip install -r requirements.txt
```

---

## How to Run

### Option 1: Google Colab

1. Open Google Colab
2. Upload the notebook or Python file
3. Run all cells

---
### Option 2: Run Using Python Terminal

1. Install the required libraries:

```bash
pip install -r requirements.txt
```

2. Run the Python file:

```bash
python wdbc_classification.py
```

For Mac/Linux users, you may need:

```bash
python3 wdbc_classification.py
```

---

### Option 3: Visual Studio Code

1. Open the project folder in VS Code
2. Install the Python extension
3. Open the Python or notebook file
4. Click **Run** or **Run All**

---

## Output Files
The program will:
- Train multiple classifiers
- Display evaluation metrics
- Generate visualization images automatically:
   - `wdbc_classification_matrix.png`
   - `wdbc_classification_scatter.png`

---

## Best Classifier Result

Based on the results, the best classifier are Decision Tree and Random Forest

Best recorded accuracy:

```text
1.0
```

---

## Library Used

- scikit-learn
- matplotlib
- numpy

---

## License

This project is licensed under the MIT License.
