import numpy as np
import pandas as pd
from scipy.signal import welch
import mne
from specparam import SpectralModel
import warnings

mne.set_log_level('ERROR')
warnings.filterwarnings('ignore')

BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 12),
    'beta': (12, 30),
    'gamma': (30, 45)
}

def bandpower(data, sf, band):
    fmin, fmax = band
    f, pxx = welch(data, fs=sf, nperseg=sf*2)
    idx = (f >= fmin) & (f <= fmax)
    return np.trapz(pxx[idx], f[idx])

def extract_features_from_raw(raw, verbose=False):
    """
    Extract features from raw EEG data.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Preprocessed raw EEG data
    verbose : bool
        Whether to print progress messages
    """
    if verbose:
        print("[EPOCHING] Creating 10-second segments...")
    epochs = mne.make_fixed_length_epochs(raw, duration=10.0, overlap=0.0, verbose='ERROR')
    sf = int(raw.info["sfreq"])

    features = []

    for ep in epochs:
        feat = {}
        for ch_idx, ch in enumerate(raw.ch_names):
            sig = ep[ch_idx, :].ravel()
            total = bandpower(sig, sf, (0.5, 45))

            for bname, br in BANDS.items():
                bp = bandpower(sig, sf, br)
                feat[f"{ch}_{bname}_abs"] = bp
                feat[f"{ch}_{bname}_rel"] = bp / total if total > 0 else 0.0

        try:
            f, Pxx = welch(ep.mean(axis=0), fs=sf, nperseg=sf*2)
            mask = (f >= 1) & (f <= 40)
            
            if np.sum(mask) > 10:  # Ensure enough frequency points
                fm = SpectralModel(peak_width_limits=[1.0, 8.0], verbose=False)
                fm.fit(f[mask], Pxx[mask], freq_range=(1, 40))

                aperiodic = fm.get_params('aperiodic')
                # aperiodic is typically [offset, exponent] or [offset, knee, exponent]
                # exponent is the last element
                feat["aperiodic_exponent"] = aperiodic[-1] if aperiodic is not None and len(aperiodic) > 0 else 0.0
            else:
                feat["aperiodic_exponent"] = 0.0
        except Exception:
            feat["aperiodic_exponent"] = 0.0

        features.append(feat)

    df = pd.DataFrame(features)
    if verbose:
        print("[FEATURES] Extracted:", df.shape)
    return df