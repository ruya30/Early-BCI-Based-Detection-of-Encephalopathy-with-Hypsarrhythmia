import pandas as pd
import numpy as np
import os
from tqdm import tqdm

from src.utils import list_edf_files, create_labels_from_seizure_files
from src.preprocess import load_and_preprocess
from src.extract_features import extract_features_from_raw
from src.train_model import train_model, print_metrics_table
from src.visualize import (
    plot_roc_curve, plot_correlation_matrices, plot_band_power_comparison,
    plot_scalp_map, plot_confusion_matrix, plot_z_scores, calculate_z_scores
)

DATA_PATH = r"C:\Users\ozdem\PycharmProjects\Rep2EarlyDiagnosis\data\raw"
RESULTS_DIR = r"C:\Users\ozdem\PycharmProjects\Rep2EarlyDiagnosis\results"
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "logs"), exist_ok=True)


def main():
    print("=" * 70)
    print("EPILEPTIC ENCEPHALOPATHY EARLY DIAGNOSIS - EEG ANALYSIS")
    print("=" * 70)

    edf_files = list_edf_files(DATA_PATH)
    print(f"\n[INFO] Found {len(edf_files)} EDF files")

    all_features = []
    file_labels = []

    MAX_FILES = 5
    if MAX_FILES:
        edf_files = edf_files[:MAX_FILES]
        print(f"[INFO] Processing first {len(edf_files)} files for speed (set MAX_FILES=None for all)")

    print("\n[FEATURE EXTRACTION] Processing EDF files...")
    # Use ICA=False for much faster processing (less accurate but faster)
    USE_ICA = False  # Set to True for better artifact removal (slower)
    
    for edf in tqdm(edf_files, desc="Processing files"):
        try:
            raw = load_and_preprocess(edf, use_ica=USE_ICA, verbose=False)
            df_feat = extract_features_from_raw(raw, verbose=False)
            all_features.append(df_feat)

            seizure_file = edf + ".seizures"
            has_seizure = os.path.exists(seizure_file)
            file_label = 1 if has_seizure else 0
            file_labels.extend([file_label] * len(df_feat))

        except Exception as e:
            print(f"\n[ERROR] Failed to process {edf}: {e}")
            continue

    if not all_features:
        print("[ERROR] No features extracted. Exiting.")
        return

    df_all = pd.concat(all_features, axis=0).reset_index(drop=True)
    print(f"\n[INFO] Final feature table shape: {df_all.shape}")
    print(f"[INFO] Features: {df_all.shape[1]}, Samples: {df_all.shape[0]}")

    labels = np.array(file_labels)
    n_pathological = np.sum(labels == 1)
    n_control = np.sum(labels == 0)
    print(f"\n[INFO] Label distribution:")
    print(f"  - Pathological (Ictal): {n_pathological} samples ({n_pathological / len(labels) * 100:.1f}%)")
    print(f"  - Control (Interictal): {n_control} samples ({n_control / len(labels) * 100:.1f}%)")

    if n_pathological == 0 or n_control == 0:
        print("\n[WARNING] Only one class present. Cannot train binary classifier.")
        print("Using dummy labels for demonstration.")
        # Create balanced dummy labels for demonstration
        n_samples = len(df_all)
        labels = np.zeros(n_samples)
        labels[:n_samples // 2] = 1
        np.random.shuffle(labels)
        n_pathological = np.sum(labels == 1)
        n_control = np.sum(labels == 0)
        print(f" - Pathological (dummy): {n_pathological} samples")
        print(f" - Control (dummy): {n_control} samples")

    # Train model with test data for ROC curve
    print("\n[MODEL TRAINING] Training MLP classifier.")
    results = train_model(df_all, labels, return_test_data=True)

    # Print metrics table
    metrics_df = print_metrics_table(results)

    # Save metrics to CSV
    metrics_path = os.path.join(RESULTS_DIR, "logs", "metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"[SAVED] Metrics table: {metrics_path}")

    # Split data for visualization
    df_pathological = df_all[labels == 1].copy()
    df_control = df_all[labels == 0].copy()

    eeg_columns = [col for col in df_all.columns if
                   any(band in col for band in ['delta', 'theta', 'alpha', 'beta', 'gamma'])]

    print(f"\n[VISUALIZATION] Generating plots (will display on screen)...")

    # 1. ROC Curve
    if 'test_data' in results:
        plot_roc_curve(
            results['test_data']['y_test'],
            results['test_data']['y_probs'],
            save_path=os.path.join(FIGURES_DIR, "roc_curve.png")
        )

        # Confusion Matrix
        plot_confusion_matrix(
            results['test_data']['y_test'],
            results['test_data']['y_pred'],
            save_path=os.path.join(FIGURES_DIR, "confusion_matrix.png")
        )

    # 2. Correlation Matrices
    if len(df_pathological) > 0 and len(df_control) > 0:
        plot_correlation_matrices(
            df_pathological, df_control, eeg_columns,
            save_dir=FIGURES_DIR
        )

        # 3. Band Power Comparisons
        # Get unique channels from column names
        channels = list(set([col.split('_')[0] for col in eeg_columns if '_' in col]))

        for band in ['theta', 'alpha', 'beta', 'gamma']:
            plot_band_power_comparison(
                df_pathological, df_control, band, channels,
                save_dir=FIGURES_DIR
            )

        # 4. Z-Score Analysis
        z_scores = calculate_z_scores(df_pathological, df_control, eeg_columns)
        if z_scores:
            plot_z_scores(
                z_scores, top_n=30,
                save_path=os.path.join(FIGURES_DIR, "z_scores.png")
            )

            # Save Z-scores to CSV
            z_df = pd.DataFrame(list(z_scores.items()), columns=['Feature', 'Z_Score'])
            z_df = z_df.sort_values('Z_Score', key=abs, ascending=False)
            z_path = os.path.join(RESULTS_DIR, "logs", "z_scores.csv")
            z_df.to_csv(z_path, index=False)
            print(f"[SAVED] Z-scores: {z_path}")

        # 5. Scalp Maps (if we have standard channel names)
        # This requires channel position mapping - simplified version
        theta_cols = [col for col in eeg_columns if 'theta' in col and '_abs' in col]
        if theta_cols:
            # Extract channel names
            channels = [col.split('_')[0] for col in theta_cols]

            # Simple channel positions (approximate 10-20 system)
            # This is a simplified mapping - adjust based on your actual channels
            channel_positions = {}
            for i, ch in enumerate(channels):
                # Approximate positions (normalized)
                angle = 2 * np.pi * i / len(channels)
                x = 0.7 * np.cos(angle)
                y = 0.7 * np.sin(angle)
                channel_positions[ch] = (x, y)

            # Calculate mean theta power
            path_theta = df_pathological[theta_cols].mean()
            control_theta = df_control[theta_cols].mean()

            # Create index for plotting
            path_theta.index = channels
            control_theta.index = channels

            plot_scalp_map(
                control_theta, channels, channel_positions,
                "Topographic Theta Power – Control Group",
                save_path=os.path.join(FIGURES_DIR, "scalp_map_control.png")
            )

            plot_scalp_map(
                path_theta, channels, channel_positions,
                "Topographic Theta Power – Pathological Group",
                save_path=os.path.join(FIGURES_DIR, "scalp_map_pathological.png")
            )

    # Save feature dataframe
    df_all['label'] = labels
    features_path = os.path.join(RESULTS_DIR, "logs", "features.csv")
    df_all.to_csv(features_path, index=False)
    print(f"[SAVED] Feature matrix: {features_path}")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print(f"Results saved to: {RESULTS_DIR}")
    print(f"Figures saved to: {FIGURES_DIR}")
    print("=" * 70)

if __name__ == "__main__":
    main()