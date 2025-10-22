# Maternal Health Risk Classification

This project builds and evaluates multiple machine learning models to predict **maternal health risk levels** (low, mid, high) using clinical and demographic data from the `maternal.csv` dataset. It includes model training, performance evaluation, and visualization outputs such as **confusion matrices** and **ROC curves**, all automatically saved to organized directories.

## 📊 Dataset Overview
- Source: [Maternal Health Risk Dataset (UCI)](https://archive.ics.uci.edu/dataset/863/maternal+health+risk)
- Observations: 1013
- Features: 6 (Age, SystolicBP, DiastolicBP, BS, BodyTemp, HeartRate)
- Target: RiskLevel (categorical: low, mid, high)

The target variable `RiskLevel`, categorized as:
- Low risk → 0  
- Mid risk → 1  
- High risk → 2


## ⚙️ Workflow Summary
1. **Data Preprocessing**
   - Loads `maternal.csv`
   - Encodes `RiskLevel`
   - Splits data into train/test sets (80/20)
   - Scales features using `StandardScaler` 

2. **Performance Evaluation Functions**
   - Computes metrics:
     - Accuracy  
     - Precision (macro)  
     - Recall (macro)  
     - F1-score (macro)  
     - ROC AUC (macro)
   - Saves:
     - Classification reports → `classification/`
     - Metric summaries → `metrics/`
     - Confusion matrices → `confusion_matrix/`
     - ROC curves → `roc/`
     - Train/test datasets → `datasets/`

3. **Model Training**
   - Trains 7 models:
     - Random Forest  
     - Gradient Boosting  
     - XGBoost  
     - LightGBM  
     - CatBoost  
     - Neural Network (MLP)  
     - Logistic Regression  

## 📁 Output Structure
After running the script, the following folders will be created:

datasets/
├── random_forest_X_train.csv
├── random_forest_X_test.csv
├── random_forest_y_train.csv
├── random_forest_y_test.csv
└── ... (other models)

metrics/
├── Random_Forest_metrics.csv
├── Gradient_Boosting_metrics.csv
├── XGBoost_metrics.csv
├── LightGBM_metrics.csv
├── CatBoost_metrics.csv
├── Neural_Network_metrics.csv
└── Logistic_Regression_metrics.csv

classification/
├── Random_Forest_classification_report.csv
├── Gradient_Boosting_classification_report.csv
├── XGBoost_classification_report.csv
├── LightGBM_classification_report.csv
├── CatBoost_classification_report.csv
├── Neural_Network_classification_report.csv
└── Logistic_Regression_classification_report.csv

confusion_matrix/
├── Random_Forest_confusion_matrix.csv
├── Random_Forest_confusion_matrix.png
├── Gradient_Boosting_confusion_matrix.csv
├── Gradient_Boosting_confusion_matrix.png
├── XGBoost_confusion_matrix.csv
├── XGBoost_confusion_matrix.png
├── LightGBM_confusion_matrix.csv
├── LightGBM_confusion_matrix.png
├── CatBoost_confusion_matrix.csv
├── CatBoost_confusion_matrix.png
├── Neural_Network_confusion_matrix.csv
├── Neural_Network_confusion_matrix.png
├── Logistic_Regression_confusion_matrix.csv
└── Logistic_Regression_confusion_matrix.png

roc/
├── Random_Forest_ROC_curve.png
├── Gradient_Boosting_ROC_curve.png
├── XGBoost_ROC_curve.png
├── LightGBM_ROC_curve.png
├── CatBoost_ROC_curve.png
├── Neural_Network_ROC_curve.png
└── Logistic_Regression_ROC_curve.png

## 🧩 Key Functions
### `save_datasets(X_train, X_test, y_train, y_test, model_name)`
Saves train/test splits as CSVs in the `datasets/` folder.

### `evaluate_and_save_results(model, X_train, X_test, y_train, y_test, model_name)`
Computes and prints performance metrics, then saves:
- `metrics/{model_name}_metrics.csv`
- `classification/{model_name}_classification_report.csv`

### `plot_confusion_matrix(y_true, y_pred, model_name)`
Creates a confusion matrix heatmap and saves:
- `confusion_matrix/{model_name}_confusion_matrix.png`

### `plot_roc_curve(y_true, y_proba, model_name)`
Plots the ROC curve with AUC and saves:
- `roc/{model_name}_ROC_curve.png`

## 📈 Visualizations
- **Confusion Matrix (Heatmap)**: Displays model performance across the three risk categories; saved as `.png`.
- **ROC Curve**: Plots True Positive Rate vs. False Positive Rate and includes the AUC score; saved as `.png`.

## 🧠 Models Implemented
| Model | Library | Description |
|-------|----------|-------------|
| Random Forest | sklearn.ensemble | Strong baseline for tabular data |
| Gradient Boosting | sklearn.ensemble | Sequential tree boosting |
| XGBoost | xgboost | Efficient gradient boosting implementation |
| LightGBM | lightgbm | Fast, memory-efficient tree boosting |
| CatBoost | catboost | Handles categorical data well |
| Neural Network (MLP) | sklearn.neural_network | Simple feedforward neural network |
| Logistic Regression | sklearn.linear_model | Interpretable baseline model |

## Key Findings
- **Top-performing model:** XGBoost
  - Highest accuracy, precision, macro F1-score, and ROC AUC
  - ROC AUC = 0.9379, indicating strong classification performance across all classes
- Models like LightGBM also performed well but XGBoost showed more consistent results across metrics

## Public Health Insights
- Early identification of high-risk individuals enables timely medical interventions
- Demonstrates the potential of routine clinical measurements for risk prediction
- Applicable in low-resource settings for maternal health monitoring

## Caveats & Alternatives
- Accuracy depends on data quality; dataset may not fully represent the population
- Missing social determinants of health (e.g., socio-economic status, access to healthcare) could affect predictions
- Future improvements could include:
  - Feature engineering or selection
  - Ensemble methods combining multiple models
  - Larger datasets with more variables
    
## 🚀 How to Run
1. Place your dataset (`maternal.csv`) in the same directory as the script.  
2. Run the script:
   ```bash
   python3 maternal_health_models.py
All outputs will be automatically saved in their respective folders.

## 📄 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## ⚡ Requirements
Install dependencies using:
```bash
pip install -r requirements.txt
