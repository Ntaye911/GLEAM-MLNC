import numpy as np

def hamming_loss(Y_true, Y_pred):
    return np.mean(Y_true != Y_pred)

def macro_f1(Y_true, Y_pred):
    f1_scores = []
    for j in range(Y_true.shape[1]):
        tp = np.sum((Y_true[:, j] == 1) & (Y_pred[:, j] == 1))
        fp = np.sum((Y_true[:, j] == 0) & (Y_pred[:, j] == 1))
        fn = np.sum((Y_true[:, j] == 1) & (Y_pred[:, j] == 0))
        if tp + fp + fn == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2 * tp / (2 * tp + fp + fn))
    return np.mean(f1_scores)

def ranking_loss(Y_true, Y_scores):
    rl = 0.0
    for i in range(Y_true.shape[0]):
        y_i = np.where(Y_true[i] == 1)[0]
        y_bar_i = np.where(Y_true[i] == 0)[0]
        if len(y_i) == 0 or len(y_bar_i) == 0:
            continue
        
        # Count pairs where relevant label score <= irrelevant label score
        scores_rel = Y_scores[i, y_i]
        scores_irrel = Y_scores[i, y_bar_i]
        
        # Broadcast comparison
        violations = np.sum(scores_rel[:, None] <= scores_irrel[None, :])
        rl += violations / (len(y_i) * len(y_bar_i))
        
    return rl / Y_true.shape[0]

def evaluate(Y_true, Y_scores, threshold=0.5):
    Y_pred = (Y_scores >= threshold).astype(int)
    return {
        'Hamming Loss': hamming_loss(Y_true, Y_pred),
        'Macro-F1': macro_f1(Y_true, Y_pred),
        'Ranking Loss': ranking_loss(Y_true, Y_scores)
    }