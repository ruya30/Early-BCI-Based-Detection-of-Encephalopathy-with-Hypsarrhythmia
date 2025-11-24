import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline

def train_model(df_features, labels, return_test_data=False):
    """
    Train MLP classifier with cross-validation and return detailed results.
    
    Parameters
    ----------
    df_features : pd.DataFrame
        Feature matrix
    labels : np.array
        Binary labels (0=control, 1=pathological)
    return_test_data : bool
        If True, also return test set predictions for ROC curve
        
    Returns
    -------
    dict : Contains cross-validation scores and optionally test predictions
    """
    X = df_features.values
    y = labels

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        max_iter=300,
        random_state=42
    )

    pipeline = make_pipeline(StandardScaler(), model)

    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'roc_auc': 'roc_auc'
    }

    scores = cross_validate(
        pipeline, X, y, cv=cv,
        scoring=scoring,
        return_train_score=False
    )

    results = {
        'cv_scores': scores,
        'metrics_summary': {}
    }

    for metric in scoring.keys():
        test_scores = scores[f'test_{metric}']
        results['metrics_summary'][metric] = {
            'mean': np.mean(test_scores),
            'std': np.std(test_scores),
            'scores': test_scores
        }

    if return_test_data:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model.fit(X_train_scaled, y_train)
        y_probs = model.predict_proba(X_test_scaled)[:, 1]
        y_pred = model.predict(X_test_scaled)
        
        results['test_data'] = {
            'y_test': y_test,
            'y_probs': y_probs,
            'y_pred': y_pred
        }
    
    return results

def print_metrics_table(results):
    """Print formatted metrics table."""
    metrics_summary = results['metrics_summary']
    
    formatted = {
        "Metric": [],
        "Mean": [],
        "Std": []
    }
    
    for metric, stats in metrics_summary.items():
        mean = stats['mean']
        std = stats['std']
        
        if metric == 'roc_auc':
            formatted["Metric"].append(metric.upper())
            formatted["Mean"].append(f"{mean:.4f}")
            formatted["Std"].append(f"±{std:.4f}")
        else:
            formatted["Metric"].append(metric.capitalize())
            formatted["Mean"].append(f"{mean*100:.2f}%")
            formatted["Std"].append(f"±{std*100:.2f}%")
    
    df_metrics = pd.DataFrame(formatted)
    print("\n" + "="*50)
    print("Cross-Validated ANN Performance Metrics:")
    print("="*50)
    print(df_metrics.to_string(index=False))
    print("="*50 + "\n")
    
    return df_metrics
