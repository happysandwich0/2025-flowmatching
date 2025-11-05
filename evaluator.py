import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import umap.umap_ as umap
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, 
    confusion_matrix, ConfusionMatrixDisplay, 
    roc_curve, auc, precision_recall_curve, average_precision_score,
    classification_report
)
import math

def print_and_report_metrics(name, y_true, y_pred, y_prob):
    print(f"\n{'='*5} 📈 {name} Evaluation Results {'='*5}")
    print(f"  - Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"  - F1 Score : {f1_score(y_true, y_pred):.4f}")
    print(f"  - ROC AUC  : {roc_auc_score(y_true, y_prob):.4f}")
    print("\n  - Classification Report:")
    print(classification_report(y_true, y_pred, digits=4))
    print("=" * 45)

def train_xgb(X_train, y_train):
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)
    return model

def plot_data_analysis(X_minority, synthetic_X):
    n_real = X_minority.shape[0]
    n_synthetic_sample = min(n_real * 5, synthetic_X.shape[0])
    
    X_sampled_syn = synthetic_X[:n_synthetic_sample, :]

    X_all_tsne_umap = np.vstack([X_minority, X_sampled_syn])
    labels_tsne_umap = np.array([0] * n_real + [1] * n_synthetic_sample) # 0: Real, 1: Synthetic

    X_embedded = TSNE(n_components=2, random_state=42, n_jobs=-1).fit_transform(X_all_tsne_umap)

    plt.figure(figsize=(18, 5))
    plt.suptitle("t-SNE Comparison: Real vs. Synthetic Minority Samples", fontsize=16)

    for i, title in enumerate(["All Samples", "Real Minority Samples", "Synthetic Minority Samples"]):
        plt.subplot(1, 3, i + 1)
        if i == 0:
            plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=labels_tsne_umap, cmap='coolwarm', alpha=0.6)
        else:
            c = 'blue' if i == 1 else 'orange'
            mask = (labels_tsne_umap == 0) if i == 1 else (labels_tsne_umap == 1)
            plt.scatter(X_embedded[mask, 0], X_embedded[mask, 1], c=c, alpha=0.6)
        plt.title(title)
        plt.axis('off')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    reducer = umap.UMAP(n_components=2, random_state=42)
    X_umap = reducer.fit_transform(X_all_tsne_umap)

    plt.figure(figsize=(18, 5))
    plt.suptitle("UMAP Comparison: Real vs. Synthetic Minority Samples", fontsize=16)

    for i, title in enumerate(["All Samples", "Real Minority Samples", "Synthetic Minority Samples"]):
        plt.subplot(1, 3, i + 1)
        if i == 0:
            plt.scatter(X_umap[:, 0], X_umap[:, 1], c=labels_tsne_umap, cmap='coolwarm', alpha=0.6)
        else:
            c = 'blue' if i == 1 else 'orange'
            mask = (labels_tsne_umap == 0) if i == 1 else (labels_tsne_umap == 1)
            plt.scatter(X_umap[mask, 0], X_umap[mask, 1], c=c, alpha=0.6)
        plt.title(title)
        plt.axis('off')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    n_features_plot = X_minority.shape[1]
    n_cols_kde = 5
    n_rows_kde = math.ceil(n_features_plot / n_cols_kde)

    plt.figure(figsize=(n_cols_kde * 4, n_rows_kde * 3))
    plt.suptitle("Feature Distribution Comparison: Real vs. Synthetic", fontsize=16, y=1.02)

    for i in range(n_features_plot):
        plt.subplot(n_rows_kde, n_cols_kde, i + 1)
        sns.kdeplot(X_minority[:, i], label='Real', fill=True, color='blue', alpha=0.5)
        sns.kdeplot(synthetic_X[:, i], label='Synthetic', fill=True, linestyle='--', color='orange', alpha=0.5)
        plt.title(f'Feature {i}', fontsize=10)
        plt.xticks([])
        plt.yticks([])
        if i == 0:
            plt.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()


def plot_model_comparison(y_test, xgb_prob_org, xgb_prob_new, xgb_pred_org, xgb_pred_new):
    fpr_org, tpr_org, _ = roc_curve(y_test, xgb_prob_org)
    fpr_new, tpr_new, _ = roc_curve(y_test, xgb_prob_new)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr_org, tpr_org, label=f"Original ROC (AUC = {auc(fpr_org, tpr_org):.4f})")
    plt.plot(fpr_new, tpr_new, label=f"With Synthetic ROC (AUC = {auc(fpr_new, tpr_new):.4f})")

    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    prec_org, recall_org, _ = precision_recall_curve(y_test, xgb_prob_org)
    prec_new, recall_new, _ = precision_recall_curve(y_test, xgb_prob_new)

    plt.figure(figsize=(8, 6))
    plt.plot(recall_org, prec_org, label=f"Original (AP = {average_precision_score(y_test, xgb_prob_org):.4f})")
    plt.plot(recall_new, prec_new, label=f"With Synthetic (AP = {average_precision_score(y_test, xgb_prob_new):.4f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    cm_org = confusion_matrix(y_test, xgb_pred_org)
    cm_new = confusion_matrix(y_test, xgb_pred_new)

    disp1 = ConfusionMatrixDisplay(confusion_matrix=cm_org)
    disp1.plot(ax=axes[0], colorbar=False)
    axes[0].set_title("Original Model")

    disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_new)
    disp2.plot(ax=axes[1], colorbar=False)
    axes[1].set_title("Model with Synthetic Data")

    plt.tight_layout()
    plt.show()