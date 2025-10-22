#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 17:30:34 2025

@author: paigezeltzer
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, roc_auc_score, confusion_matrix
)

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Load the data
os.getcwd()
data = pd.read_csv('maternal.csv')
risk_mapping = {'low risk': 0, 'mid risk': 1, 'high risk': 2}
data['RiskLevel_encoded'] = data['RiskLevel'].map(risk_mapping)

X = data.drop(['RiskLevel', 'RiskLevel_encoded'], axis=1)
y = data['RiskLevel_encoded']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#save train and test datasets
def save_datasets(X_train, X_test, y_train, y_test, model_name):
    if not os.path.exists('datasets'):
        os.makedirs('datasets')
    pd.DataFrame(X_train).to_csv(f'datasets/{model_name}_X_train.csv', index=False)
    pd.DataFrame(X_test).to_csv(f'datasets/{model_name}_X_test.csv', index=False)
    pd.DataFrame(y_train).to_csv(f'datasets/{model_name}_y_train.csv', index=False)
    pd.DataFrame(y_test).to_csv(f'datasets/{model_name}_y_test.csv', index=False)


#performance metrics
def evaluate_and_save_results(model, X_train, X_test, y_train, y_test, model_name):
    try:
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        y_train_proba = model.predict_proba(X_train)
        y_test_proba = model.predict_proba(X_test)
        
        # Compute metrics
        metrics = {
            "Train Accuracy": accuracy_score(y_train, y_train_pred),
            "Test Accuracy": accuracy_score(y_test, y_test_pred),
            "Train Precision (macro)": precision_score(y_train, y_train_pred, average='macro'),
            "Test Precision (macro)": precision_score(y_test, y_test_pred, average='macro'),
            "Train Recall (macro)": recall_score(y_train, y_train_pred, average='macro'),
            "Test Recall (macro)": recall_score(y_test, y_test_pred, average='macro'),
            "Train F1-score (macro)": f1_score(y_train, y_train_pred, average='macro'),
            "Test F1-score (macro)": f1_score(y_test, y_test_pred, average='macro'),
            "Train ROC AUC (macro)": roc_auc_score(y_train, y_train_proba, multi_class="ovr", average="macro"),
            "Test ROC AUC (macro)": roc_auc_score(y_test, y_test_proba, multi_class="ovr", average="macro")
        }
        
        # Print metrics
        print(f"{model_name} Classification Performance Metrics:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
        print("\nClassification Report:")
        report = classification_report(y_test, y_test_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        print(report_df)
        print("-" * 50)
        
        #create directory for metrics if it doesn't exist
        if not os.path.exists('metrics'):
            os.makedirs('metrics')
        # Save metrics summary
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(f'metrics/{model_name}_metrics.csv', index=False)
        
        #create directory for classification if it doesn't exist
        if not os.path.exists('classification'):
            os.makedirs('classification')
        # Save classification report
        report_df.to_csv(f'classification/{model_name}_classification_report.csv')
        
        print(f"Metrics saved as metrics/{model_name}_metrics.csv")
        print(f"Classification report saved as classification/{model_name}_classification_report.csv")
        
        return metrics_df, report_df
    
    except Exception as e:
        print(f"Error occurred: {e}")
        return None, None

#confusion matrix
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, model_name):
    # Create directory for figures if it doesn't exist
    if not os.path.exists('confusion_matrix'):
        os.makedirs('confusion_matrix')
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
   
    # Plot confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'{model_name} - Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Save plot as image
    filepath = f'confusion_matrix/{model_name}_confusion_matrix.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix image saved as confusion_matrix/{model_name}_confusion_matrix.png ")
    
    plt.show()
    
    return cm



#roc curve
from sklearn.metrics import roc_curve, auc
import os

def plot_roc_curve(y_true, y_proba, model_name):
    # Create directory for roc curve if it doesn't exist
    if not os.path.exists('roc'):
        os.makedirs('roc')
    
    # Compute ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_proba[:, 1], pos_label=1)  # class 1 as positive
    roc_auc = auc(fpr, tpr)
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'{model_name} (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Random classifier
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc='lower right')
    
    # Save figure
    filepath = f'roc/{model_name}_ROC_curve.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"ROC curve saved as roc/{model_name}_ROC_curve.png")
    
    plt.show()


# 1. Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
save_datasets(X_train, X_test, y_train, y_test, "random_forest")
evaluate_and_save_results(rf, X_train, X_test, y_train, y_test, "Random_Forest")
plot_confusion_matrix(y_test, rf_pred, "Random Forest")
plot_roc_curve(y_test, rf.predict_proba(X_test), "Random Forest")

# 2. Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb.fit(X_train, y_train)
gb_pred = gb.predict(X_test)
save_datasets(X_train, X_test, y_train, y_test, "gradient_boosting")
evaluate_and_save_results(gb, X_train, X_test, y_train, y_test, "Gradient_Boosting")
plot_confusion_matrix(y_test, gb_pred, "Gradient Boosting")
plot_roc_curve(y_test, gb.predict_proba(X_test), "Gradient Boosting")

# 3. XGBoost
xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)
save_datasets(X_train, X_test, y_train, y_test, "xgboost")
evaluate_and_save_results(xgb, X_train, X_test, y_train, y_test, "XGBoost")
plot_confusion_matrix(y_test, xgb_pred, "XGBoost")
plot_roc_curve(y_test, xgb.predict_proba(X_test), "XGBoost")

# 4. LightGBM
lgb = LGBMClassifier(random_state=42)
lgb.fit(X_train, y_train)
lgb_pred = lgb.predict(X_test)
save_datasets(X_train, X_test, y_train, y_test, "lightgbm")
evaluate_and_save_results(lgb, X_train, X_test, y_train, y_test, "LightGBM")
plot_confusion_matrix(y_test, lgb_pred, "LightGBM")
plot_roc_curve(y_test, lgb.predict_proba(X_test), "LightGBM")

# 5. CatBoost
cat = CatBoostClassifier(random_state=42, verbose=0)
cat.fit(X_train, y_train)
cat_pred = cat.predict(X_test)
save_datasets(X_train, X_test, y_train, y_test, "catboost")
evaluate_and_save_results(cat, X_train, X_test, y_train, y_test, "CatBoost")
plot_confusion_matrix(y_test, cat_pred, "CatBoost")
plot_roc_curve(y_test, cat.predict_proba(X_test), "CatBoost")

# 6. Neural Network
nn = MLPClassifier(max_iter=1000, random_state=42)
nn.fit(X_train_scaled, y_train)
nn_pred = nn.predict(X_test_scaled)
save_datasets(X_train_scaled, X_test_scaled, y_train, y_test, "neural_network")
evaluate_and_save_results(nn, X_train_scaled, X_test_scaled, y_train, y_test, "Neural_Network")
plot_confusion_matrix(y_test, nn_pred, "Neural Network")
plot_roc_curve(y_test, nn.predict_proba(X_test_scaled), "Neural Network")

#7 Logistic Regression
logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_train_scaled, y_train)
logreg_pred = logreg.predict(X_test_scaled)
save_datasets(X_train_scaled, X_test_scaled, y_train, y_test, "logistic_regression")
evaluate_and_save_results(logreg, X_train_scaled, X_test_scaled, y_train, y_test, "Logistic Regression")
plot_confusion_matrix(y_test, logreg_pred, "Logistic Regression")
plot_roc_curve(y_test, logreg.predict_proba(X_test_scaled), "Logistic Regression")
