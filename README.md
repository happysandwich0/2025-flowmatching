# 2025-flowmatching: Addressing Class Imbalance in Credit Card Fraud Detection using Forest Flow

## 📌 Project Overview
This project investigates the effectiveness of **Forest Flow**, a Flow-based generative model, in synthesizing minority class (fraudulent) transactions to mitigate the severe class imbalance challenge in credit card fraud detection. The goal is to demonstrate that this data augmentation technique can significantly enhance the predictive performance, especially in detecting rare fraud cases.

## 📊 Key Results

### 1. Forest Flow Data Augmentation
* **Model:** Forest Flow combines the structured representation of XGBoost with the expressive power of Continuous Normalizing Flows to generate realistic synthetic tabular data.
* **Data Generation:** Used the Kaggle credit card fraud dataset to generate **14,000** synthetic fraudulent transactions.
* **Fidelity Check:** t-SNE and UMAP visualizations confirmed that the synthetic samples closely approximate the distribution and structure of the real fraud data.

### 2. Classification Performance Comparison
* **Classifier:** An XGBoost model was trained on the original dataset (`xgb_org`) and the augmented dataset (`xgb_new`).
* **Results:** The augmented model showed significant performance improvements in critical metrics for fraud detection on the **test set**:
    * **ROC AUC:** Improved from **0.9390** to **0.9962**.
    * **Recall:** Improved from **0.7959** to **0.8571**.

## 💾 Dataset
* **Source:** [Credit Card Fraud Detection Dataset on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
* **Imbalance:** The minority class (fraud) represents only **0.172%** of the total observations, highlighting the severity of the class imbalance.

## 🔗 References
* **Forest Diffusion Repository:** [SamsungSAILMontreal/ForestDiffusion](https://github.com/SamsungSAILMontreal/ForestDiffusion)
