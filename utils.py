import os
import numpy as np

def list_edf_files(data_raw_folder):
    return [
        os.path.join(data_raw_folder, f)
        for f in os.listdir(data_raw_folder)
        if f.endswith(".edf")
    ]

def create_labels_from_seizure_files(edf_files, epoch_duration=10.0):
    """
    Create binary labels based on seizure file presence.
    
    Parameters
    ----------
    edf_files : list
        List of EDF file paths
    epoch_duration : float
        Duration of each epoch in seconds
        
    Returns
    -------
    labels : np.array
        Binary labels (0=interictal/control, 1=ictal/pathological)
    """
    labels = []
    
    for edf_file in edf_files:
        seizure_file = edf_file + ".seizures"
        has_seizure = os.path.exists(seizure_file)
        
        # For now, if seizure file exists, mark all epochs as pathological
        # In future, can parse seizure times to label specific epochs
        # For simplicity: file with seizures = pathological (1), without = control (0)
        label = 1 if has_seizure else 0
        
        # Estimate number of epochs (will be adjusted in main)
        # This is a placeholder - actual epoch count comes from feature extraction
        labels.append(label)
    
    return np.array(labels)