import mne
import warnings

mne.set_log_level('ERROR')
warnings.filterwarnings('ignore', category=RuntimeWarning)

def load_and_preprocess(edf_path, use_ica=True, verbose=False):
    """
    Load and preprocess EDF file.
    
    Parameters
    ----------
    edf_path : str
        Path to EDF file
    use_ica : bool
        Whether to apply ICA (slower but better artifact removal)
    verbose : bool
        Whether to print progress messages
    """
    if verbose:
        print(f"\n[LOAD] {edf_path}")

    mne.set_log_level('ERROR')
    
    raw = mne.io.read_raw_edf(edf_path, preload=True, stim_channel=None, verbose=False)

    raw.filter(l_freq=0.5, h_freq=45.0, fir_design='firwin', verbose=False)
    raw.notch_filter(freqs=[50.0], verbose=False)

    raw.set_eeg_reference('average', projection=False, verbose=False)

    if use_ica:
        n_components = min(15, len(raw.ch_names) - 1)
        ica = mne.preprocessing.ICA(n_components=n_components, random_state=97, 
                                    max_iter=200, verbose=False)
        ica.fit(raw, verbose=False)
        ica.exclude = []
        raw_clean = ica.apply(raw.copy(), verbose=False)
    else:
        raw_clean = raw

    return raw_clean
