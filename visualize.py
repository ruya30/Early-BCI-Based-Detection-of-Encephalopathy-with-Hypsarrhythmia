import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
from scipy.interpolate import griddata
import os

try:
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
    try:
        plt.style.use('seaborn-whitegrid')
    except OSError:
        plt.style.use('seaborn-white')
sns.set_palette("Set2")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

def plot_roc_curve(y_test, y_probs, save_path=None):
    """Plot ROC curve for binary classification."""
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(9, 7))
    plt.plot(fpr, tpr, color='#8B4CBF', lw=3, 
             label=f'ROC curve (AUC = {roc_auc:.3f})', marker='o', markersize=4, markevery=0.1)
    plt.plot([0, 1], [0, 1], color='#95A5A6', lw=2.5, linestyle=':', 
             label='Random classifier', alpha=0.7)
    plt.fill_between(fpr, tpr, alpha=0.2, color='#8B4CBF')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=13, fontweight='medium')
    plt.ylabel('True Positive Rate', fontsize=13, fontweight='medium')
    plt.title('ROC Curve - Epileptic Encephalopathy Detection', fontsize=15, fontweight='bold', pad=15)
    plt.legend(loc="lower right", fontsize=12, framealpha=0.95, shadow=True)
    plt.grid(True, alpha=0.25, linestyle='--', linewidth=0.8)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"[SAVED] ROC curve: {save_path}")
    plt.show()

def plot_correlation_matrices(df_pathological, df_control, eeg_columns, save_dir=None):
    """Plot correlation matrices for pathological and control groups."""

    plt.figure(figsize=(15, 13))
    corr_path = df_pathological[eeg_columns].corr()
    sns.heatmap(corr_path, cmap='RdYlBu_r', vmin=-1, vmax=1, center=0,
                square=True, cbar_kws={"shrink": 0.75, "label": "Correlation Coefficient"},
                xticklabels=False, yticklabels=False, linewidths=0.5, linecolor='white')
    plt.title('Correlation Matrix – Pathological Group (Epileptic Encephalopathy)', 
              fontsize=15, fontweight='bold', pad=25)
    plt.tight_layout()
    
    if save_dir:
        path_save = os.path.join(save_dir, 'correlation_matrix_pathological.png')
        plt.savefig(path_save, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"[SAVED] Correlation matrix (pathological): {path_save}")
    plt.show()

    plt.figure(figsize=(15, 13))
    corr_control = df_control[eeg_columns].corr()
    sns.heatmap(corr_control, cmap='RdYlBu_r', vmin=-1, vmax=1, center=0,
                square=True, cbar_kws={"shrink": 0.75, "label": "Correlation Coefficient"},
                xticklabels=False, yticklabels=False, linewidths=0.5, linecolor='white')
    plt.title('Correlation Matrix – Control Group (Interictal)', 
              fontsize=15, fontweight='bold', pad=25)
    plt.tight_layout()
    
    if save_dir:
        control_save = os.path.join(save_dir, 'correlation_matrix_control.png')
        plt.savefig(control_save, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"[SAVED] Correlation matrix (control): {control_save}")
    plt.show()

def plot_band_power_comparison(df_pathological, df_control, band_name, channels, save_dir=None):
    """Plot band power comparison between groups."""

    path_cols = [f"{ch}_{band_name}_abs" for ch in channels if f"{ch}_{band_name}_abs" in df_pathological.columns]
    control_cols = [f"{ch}_{band_name}_abs" for ch in channels if f"{ch}_{band_name}_abs" in df_control.columns]
    
    if not path_cols or not control_cols:
        print(f"[WARNING] No {band_name} columns found for comparison")
        return
    
    path_power = df_pathological[path_cols].mean()
    control_power = df_control[control_cols].mean()

    x = np.arange(len(path_cols))
    width = 0.38
    
    plt.figure(figsize=(15, 7))
    bars1 = plt.bar(x - width/2, path_power.values, width, label='Pathological', 
            color='#C0392B', alpha=0.85, edgecolor='#922B21', linewidth=1.2)
    bars2 = plt.bar(x + width/2, control_power.values, width, label='Control', 
            color='#27AE60', alpha=0.85, edgecolor='#1E8449', linewidth=1.2)

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=7, rotation=90)
    
    plt.xlabel('Channels', fontsize=13, fontweight='medium')
    plt.ylabel(f'{band_name.capitalize()} Power (μV²/Hz)', fontsize=13, fontweight='medium')
    plt.title(f'Mean {band_name.capitalize()} Power Comparison: Pathological vs Control', 
              fontsize=15, fontweight='bold', pad=15)
    plt.xticks(x, [col.replace(f'_{band_name}_abs', '') for col in path_cols], 
               rotation=50, ha='right', fontsize=10)
    plt.legend(fontsize=12, framealpha=0.95, shadow=True, loc='upper right')
    plt.grid(True, alpha=0.25, axis='y', linestyle='--', linewidth=0.8)
    plt.tight_layout()
    
    if save_dir:
        save_path = os.path.join(save_dir, f'{band_name}_power_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"[SAVED] {band_name} power comparison: {save_path}")
    plt.show()

def plot_scalp_map(theta_values, channels, channel_positions, title, save_path=None):
    """Plot topographic scalp map for theta power."""

    x, y, v = [], [], []
    for ch in channels:
        if ch in channel_positions and ch in theta_values.index:
            xi, yi = channel_positions[ch]
            x.append(xi)
            y.append(yi)
            v.append(theta_values[ch])
    
    if not x:
        print(f"[WARNING] No valid channel data for scalp map")
        return

    grid_x, grid_y = np.mgrid[-1.2:1.2:300j, -1.2:1.2:300j]
    grid_z = griddata((x, y), v, (grid_x, grid_y), method='cubic', fill_value=np.nan)

    plt.figure(figsize=(9, 9))
    contour = plt.contourf(grid_x, grid_y, grid_z, levels=60, cmap='viridis', 
                           vmin=min(v), vmax=max(v), alpha=0.9)
    plt.colorbar(contour, label='Power (μV²/Hz)', shrink=0.75, pad=0.02)

    head = plt.Circle((0, 0), 1.05, color='#2C3E50', fill=False, linewidth=3.5, zorder=4)
    plt.gca().add_patch(head)

    nose_x = [0, -0.1, 0.1]
    nose_y = [1.05, 0.95, 0.95]
    plt.plot(nose_x, nose_y, color='#2C3E50', linewidth=3, zorder=4)

    plt.scatter(x, y, c='#ECF0F1', s=120, edgecolors='#34495E', linewidths=2, 
                zorder=5, alpha=0.9)

    for ch, (xi, yi) in channel_positions.items():
        if ch in theta_values.index:
            plt.text(xi, yi, ch, ha='center', va='center', fontsize=9, 
                    color='#2C3E50', weight='bold', zorder=6, 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                             edgecolor='#34495E', alpha=0.7, linewidth=1))
    
    plt.title(title, fontsize=15, fontweight='bold', pad=25)
    plt.axis('equal')
    plt.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"[SAVED] Scalp map: {save_path}")
    plt.show()

def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', 
                xticklabels=['Control', 'Pathological'],
                yticklabels=['Control', 'Pathological'],
                cbar_kws={"label": "Count", "shrink": 0.8},
                linewidths=2, linecolor='white', square=True,
                annot_kws={'size': 14, 'weight': 'bold', 'color': 'white'})
    plt.ylabel('True Label', fontsize=13, fontweight='medium')
    plt.xlabel('Predicted Label', fontsize=13, fontweight='medium')
    plt.title('Confusion Matrix', fontsize=15, fontweight='bold', pad=15)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"[SAVED] Confusion matrix: {save_path}")
    plt.show()

def plot_feature_importance(feature_names, importances, top_n=20, save_path=None):
    """Plot feature importance (top N features)."""

    indices = np.argsort(importances)[::-1][:top_n]
    
    plt.figure(figsize=(11, 9))
    plt.barh(range(top_n), importances[indices], color='#9B59B6', alpha=0.85, 
            edgecolor='#7D3C98', linewidth=1.5)
    plt.yticks(range(top_n), [feature_names[i] for i in indices], fontsize=10)
    plt.xlabel('Importance', fontsize=13, fontweight='medium')
    plt.title(f'Top {top_n} Most Important Features', fontsize=15, fontweight='bold', pad=15)
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.25, axis='x', linestyle='--', linewidth=0.8)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"[SAVED] Feature importance: {save_path}")
    plt.show()

def calculate_z_scores(df_pathological, df_control, columns):
    """Calculate Z-scores for features."""
    z_scores = {}
    
    for col in columns:
        if col in df_pathological.columns and col in df_control.columns:
            path_mean = df_pathological[col].mean()
            control_mean = df_control[col].mean()
            control_std = df_control[col].std()
            
            if control_std > 0:
                z_score = (path_mean - control_mean) / control_std
                z_scores[col] = z_score
    
    return z_scores

def plot_z_scores(z_scores, top_n=30, save_path=None):
    """Plot Z-scores for top features."""

    sorted_scores = sorted(z_scores.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    features, scores = zip(*sorted_scores)
    
    colors = ['#E67E22' if s > 0 else '#3498DB' for s in scores]
    
    plt.figure(figsize=(13, 9))
    bars = plt.barh(range(len(features)), scores, color=colors, alpha=0.85, 
                    edgecolor='white', linewidth=1.5)
    plt.yticks(range(len(features)), features, fontsize=10)
    plt.xlabel('Z-Score', fontsize=13, fontweight='medium')
    plt.title(f'Top {top_n} Features by Z-Score (Pathological vs Control)', 
              fontsize=15, fontweight='bold', pad=15)
    plt.axvline(x=0, color='#34495E', linestyle='-', linewidth=2, alpha=0.7)
    plt.axvline(x=2, color='#E74C3C', linestyle='--', linewidth=2, alpha=0.6, label='±2 SD')
    plt.axvline(x=-2, color='#E74C3C', linestyle='--', linewidth=2, alpha=0.6)
    plt.legend(fontsize=11, framealpha=0.95, shadow=True)
    plt.grid(True, alpha=0.25, axis='x', linestyle='--', linewidth=0.8)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"[SAVED] Z-scores plot: {save_path}")
    plt.show()

