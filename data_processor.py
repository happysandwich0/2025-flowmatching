import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from ForestDiffusion import ForestDiffusionModel

FILE_PATH = "creditcard.csv"

def load_and_split_data(file_path=FILE_PATH, test_size=0.2, random_state=42):
    try:
        my_data = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: {file_path} 파일 없음.")
        return None, None, None, None, None
        
    X_all, y_all = my_data.drop(columns=['Class']), my_data['Class']
    X_all = X_all.to_numpy()

    X_train_org, X_test, y_train_org, y_test = train_test_split(
        X_all, y_all, test_size=test_size, random_state=random_state, stratify=y_all)
    
    X_minority = X_all[y_all == 1]

    print(f"Train size: {len(X_train_org)}, Test size: {len(X_test)}")
    print(f"Real Minority Samples: {X_minority.shape[0]}")
    
    return X_train_org, y_train_org, X_test, y_test, X_minority


def generate_synthetic_data(X_minority, batch_size=100000, n_t=30):
    model = ForestDiffusionModel(
        X=X_minority, 
        label_y=None, 
        diffusion_type='flow',
        n_t=30,
        duplicate_K=50
    )

    synthetic_X = model.generate(batch_size=batch_size, n_t=n_t)
    print(f"Synthetic data generated: {synthetic_X.shape}")
    
    return synthetic_X