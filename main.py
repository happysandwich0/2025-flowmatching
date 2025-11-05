import numpy as np
from data_processor import load_and_split_data, generate_synthetic_data
from evaluator import (
    train_xgb, 
    print_and_report_metrics, 
    plot_data_analysis, 
    plot_model_comparison
)

def main():
    X_train_org, y_train_org, X_test, y_test, X_minority = load_and_split_data()
    
    if X_train_org is None:
        return

    synthetic_X = generate_synthetic_data(X_minority, batch_size=100000)

    if len(synthetic_X) > 0:
        plot_data_analysis(X_minority, synthetic_X)

    model_org = train_xgb(X_train_org, y_train_org)
    xgb_pred_org = model_org.predict(X_test)
    xgb_prob_org = model_org.predict_proba(X_test)[:, 1]
    print_and_report_metrics("Original Model (xgb_org)", y_test, xgb_pred_org, xgb_prob_org)

    X_resampled = np.vstack([X_train_org, synthetic_X])
    y_resampled = np.concatenate([y_train_org, np.ones(len(synthetic_X))])

    model_new = train_xgb(X_resampled, y_resampled)
    xgb_pred_new = model_new.predict(X_test)
    xgb_prob_new = model_new.predict_proba(X_test)[:, 1]
    print_and_report_metrics("Model with Synthetic Data (xgb_new)", y_test, xgb_pred_new, xgb_prob_new)

    plot_model_comparison(y_test, xgb_prob_org, xgb_prob_new, xgb_pred_org, xgb_pred_new)


if __name__ == "__main__":
    main()