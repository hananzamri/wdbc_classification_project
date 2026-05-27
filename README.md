# WDBC Classification Project

## Project Purpose
This project uses machine learning classification models to classify breast cancer tumors as malignant or benign using the Wisconsin Diagnostic Breast Cancer dataset.

## Dataset Used
Built-in Breast Cancer dataset from scikit-learn.

## Classifiers Tested
- Support Vector Machine (SVM)
- Decision Tree
- K-Nearest Neighbors (KNN)
- Random Forest

## Requirements Installation

Install the required libraries using:

```bash
pip install -r requirements.txt

## How to Run

### Option 1: Google Colab

1. Go to Google Colab
2. Upload the .ipynb file from this repository
3. Open the notebook
4. Click Run All

### Option 2: Jupyter Notebook

1. Install Jupyter:
   pip install notebook
2. Run:
   jupyter notebook
3. Open the `.ipynb` file and run all cells

### Option 3: Visual Studio Code

1. Open the project folder in VS Code
2. Install the Python extension (if not already installed)
3. Open main.ipynb file
4. Click Run or Run All

## Output Files
The project generate:
- wdbc_classification_matrix.png
- wdbc_classification_scatter.png

## Best Classifier Result
The best classifier based on the highest accuracy is Decision Tree and Random Forest with 1.0 accuracy