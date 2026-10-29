import numpy as np
from scipy.linalg import svd

class GLEAM:
    def __init__(self, r=8, knn=30, lambda1=0.1, lambda2=0.1, lambda3=1.0, 
                 lambda4=0.01, lambda5=0.1, lambda6=1.0, mu=0.01, rho=1.1, 
                 mu_max=1e6, max_iter=100, tol=1e-5):
        self.r = r
        self.knn = knn
        self.lambdas = (lambda1, lambda2, lambda3, lambda4, lambda5, lambda6)
        self.mu = mu
        self.rho = rho
        self.mu_max = mu_max
        self.max_iter = max_iter
        self.tol = tol

    def _svt(self, X, tau):
        """Singular Value Thresholding operator"""
        U, S, Vt = svd(X, full_matrices=False)
        S_thresh = np.maximum(S - tau, 0)
        return U @ np.diag(S_thresh) @ Vt

    def _update_S_row(self, distances, lambda6):
        """Closed-form row-wise update for adaptive similarity matrix S (Nie et al., 2014)"""
        n = len(distances)
        s = np.zeros(n)
        # Sort distances to find the optimal k
        sorted_idx = np.argsort(distances)
        sorted_d = distances[sorted_idx]
        
        # Find optimal k (simplified heuristic for the closed-form solution)
        k = min(self.knn, n - 1)
        if k < 2:
            k = 2
            
        sum_d = np.sum(sorted_d[:k])
        denominator = k * sorted_d[k] - sum_d
        
        if denominator > 1e-8:
            for j in range(k):
                val = (sorted_d[k] - sorted_d[j]) / denominator
                if val > 0:
                    s[sorted_idx[j]] = val
                    
        # Ensure sum to 1 and non-negative
        s = np.maximum(s, 0)
        if np.sum(s) > 0:
            s = s / np.sum(s)
        return s

    def fit(self, X, Y):
        n, d = X.shape
        c = Y.shape[1]
        r = self.r
        
        l1, l2, l3, l4, l5, l6 = self.lambdas
        
        # Initialization
        U = np.random.randn(d, r) * 0.01
        V = np.random.randn(c, r) * 0.01
        F = np.random.randn(r, n) * 0.01
        # Orthogonalize F initially
        U_f, _, Vt_f = svd(F, full_matrices=False)
        F = U_f @ Vt_f
        
        W = np.eye(r)
        J = np.copy(U)
        K = np.copy(V)
        Lambda_U = np.zeros((d, r))
        Lambda_V = np.zeros((c, r))
        S = np.ones((n, n)) / n
        np.fill_diagonal(S, 0)
        
        mu = self.mu
        
        for iteration in range(self.max_iter):
            U_old, V_old, F_old, S_old = np.copy(U), np.copy(V), np.copy(F), np.copy(S)
            
            # 1. Compute Laplacian L_S
            D_S = np.diag(np.sum(S, axis=1))
            L_S = D_S - S
            
            # 2. Update U
            A_U = X.T @ X + l1 * (X.T @ L_S @ X) + (mu / 2) * np.eye(d)
            B_U = X.T @ F.T + (mu / 2) * (J - Lambda_U / mu) # Note: X is (n, d), so X.T is (d, n)
            # Fix dimensions: X in paper is d x n. Let's assume X is n x d here for standard sklearn compatibility.
            # If X is n x d: X.T @ X is d x d. X.T @ F.T is d x r.
            U = np.linalg.solve(A_U, B_U)
            
            # 3. Update V
            A_V = Y.T @ Y + l2 * (Y.T @ L_S @ Y) + (mu / 2) * np.eye(c)
            B_V = Y.T @ F.T @ W.T + (mu / 2) * (K - Lambda_V / mu)
            V = np.linalg.solve(A_V, B_V)
            
            # 4. Update W
            # W = V^T Y F^T (F F^T + l5 I)^{-1}
            # Since F F^T ≈ I, this is well-conditioned
            W = (V.T @ Y.T @ F.T) @ np.linalg.inv(F @ F.T + l5 * np.eye(r))
            
            # 5. Update J and K (SVT)
            tau = l4 / mu
            J = self._svt(U + Lambda_U / mu, tau)
            K = self._svt(V + Lambda_V / mu, tau)
            
            # 6. Update S (Adaptive Graph)
            # Compute pairwise distances in projected spaces
            UX = X @ U  # (n, r)
            VY = Y @ V  # (n, r)
            
            S_new = np.zeros((n, n))
            for i in range(n):
                dist = 0.5 * np.sum((UX - UX[i])**2, axis=1) + \
                       0.5 * np.sum((VY - VY[i])**2, axis=1) + \
                       0.5 * l3 * np.sum((F.T - F.T[i])**2, axis=1)
                dist[i] = np.inf # s_ii = 0
                S_new[i, :] = self._update_S_row(dist, l6)
            
            # Symmetrize and renormalize
            S = (S_new + S_new.T) / 2
            row_sums = np.sum(S, axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1 # Avoid division by zero
            S = S / row_sums
            np.fill_diagonal(S, 0)
            
            # 7. Update F (Stiefel Manifold Projection)
            # Gradient of the F subproblem
            grad_F = 2 * (F - U.T @ X.T) + \
                     2 * (W.T @ W @ F - W.T @ V.T @ Y.T) + \
                     2 * l3 * (F @ L_S)
            
            # Gradient descent step
            F_temp = F - 0.01 * grad_F  # 0.01 is a fixed step size; backtracking can be added
            
            # Retraction onto Stiefel manifold via SVD
            U_f, _, Vt_f = svd(F_temp, full_matrices=False)
            F = U_f @ Vt_f
            
            # 8. Update Multipliers and Penalty
            Lambda_U = Lambda_U + mu * (U - J)
            Lambda_V = Lambda_V + mu * (V - K)
            mu = min(self.rho * mu, self.mu_max)
            
            # Check convergence
            diff_U = np.linalg.norm(U - U_old, 'fro') / (np.linalg.norm(U_old, 'fro') + 1e-8)
            diff_F = np.linalg.norm(F - F_old, 'fro') / (np.linalg.norm(F_old, 'fro') + 1e-8)
            if diff_U < self.tol and diff_F < self.tol:
                break
                
        self.U = U
        self.V = V
        self.W = W
        self.F = F
        return self

    def predict_scores(self, X_test):
        """Generate raw prediction scores for test data"""
        F_test = X_test @ self.U
        scores = F_test @ self.W.T @ self.V.T
        return scores