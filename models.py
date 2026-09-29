"""Model definitions and evaluation metrics."""

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def logistic():
    """
    Heavily regularized (C=0.1) on purpose. The features are collinear —
    elo_diff, netrtg_diff and form_diff all measure roughly "who is better" —
    so weak regularization gives unstable, uninterpretable coefficients.
    """
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))


def gbm():
    """
    Shallow trees, isotonic calibration. Raw boosted trees are overconfident;
    calibration matters more than accuracy if you ever compare to a betting line.
    """
    return CalibratedClassifierCV(
        HistGradientBoostingClassifier(
            max_depth=3,
            learning_rate=0.05,
            max_iter=300,
            l2_regularization=1.0,
            early_stopping=True,
        ),
        method="isotonic",
        cv=3,
    )


def all_models():
    """Fresh, unfitted instances. Call this per fold — never reuse a fitted model."""
    return {"logistic": logistic(), "gbm": gbm()}


def evaluate(y_true, prob):
    """
    acc     — how often the pick is right
    auc     — ranking quality, threshold-free
    logloss — punishes confident mistakes
    brier   — calibration; lower means stated probabilities are honest
    """
    pred = (np.asarray(prob) >= 0.5).astype(int)
    return {
        "n": len(y_true),
        "acc": accuracy_score(y_true, pred),
        "auc": roc_auc_score(y_true, prob),
        "logloss": log_loss(y_true, prob),
        "brier": brier_score_loss(y_true, prob),
    }


def coefficients(fitted_logistic, feature_names):
    """Standardized coefficients from a fitted logistic pipeline, by magnitude."""
    coefs = fitted_logistic.named_steps["logisticregression"].coef_[0]
    return sorted(zip(feature_names, coefs), key=lambda pair: -abs(pair[1]))