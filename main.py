import argparse
import numpy as np
from sklearn.model_selection import KFold
from data_loader import load_dataset
from gleam import GLEAM
from metrics import evaluate

def main():
    parser = argparse.ArgumentParser(description="Run GLEAM on a multi-label dataset")
    parser.add_argument('--dataset', type=str, default='yeast', help='Dataset name in ./datasets/')
    parser.add_argument('--r', type=int, default=8, help='Latent dimension')
    parser.add_argument('--knn', type=int, default=30, help='Number of neighbors for adaptive graph')
    parser.add_argument('--folds', type=int, default=5, help='Number of CV folds')
    args = parser.parse_args()

    print(f"Loading dataset: {args.dataset}...")
    X, Y = load_dataset(args.dataset)
    print(f"Dataset loaded. X shape: {X.shape}, Y shape: {Y.shape}")

    kf = KFold(n_splits=args.folds, shuffle=True, random_state=42)
    
    hl_scores, mf1_scores, rl_scores = [], [], []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        print(f"\n--- Fold {fold + 1}/{args.folds} ---")
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]

        # Initialize and train GLEAM
        model = GLEAM(r=args.r, knn=args.knn)
        model.fit(X_train, Y_train)

        # Predict
        Y_scores = model.predict_scores(X_test)
        
        # Evaluate (using 0.5 as default threshold, can be tuned via inner CV)
        metrics = evaluate(Y_test, Y_scores, threshold=0.5)
        
        hl_scores.append(metrics['Hamming Loss'])
        mf1_scores.append(metrics['Macro-F1'])
        rl_scores.append(metrics['Ranking Loss'])
        
        print(f"Fold {fold+1} | HL: {metrics['Hamming Loss']:.4f} | Macro-F1: {metrics['Macro-F1']:.4f} | RL: {metrics['Ranking Loss']:.4f}")

    print("\n" + "="*60)
    print("FINAL RESULTS (Mean ± Std)")
    print("="*60)
    print(f"Hamming Loss : {np.mean(hl_scores):.4f} ± {np.std(hl_scores):.4f}")
    print(f"Macro-F1     : {np.mean(mf1_scores):.4f} ± {np.std(mf1_scores):.4f}")
    print(f"Ranking Loss : {np.mean(rl_scores):.4f} ± {np.std(rl_scores):.4f}")
    print("="*60)

if __name__ == "__main__":
    main()