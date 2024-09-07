""" 
Utility functions for preparing fMRI, events, and design matrices for GLM analysis 
"""

# data_root = "/project/6019337/databases/religion/bids/"	# cedar

import os, sys, json
import numpy as np
import pandas as pd
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn import image

# Function to set duration, in seconds, based on trial type (question, image, response)
def set_event_duration(trial_type):
    if trial_type.startswith('question:'):
        return 1.0
    elif trial_type.startswith('image:'):
        return 1.0
    elif trial_type.startswith('response:'):
        return 0.0
    else:
        return 0.0  # Default value if none of the conditions match


def make_localizer_contrasts(design_matrix):
    """
    Return a dictionary of contrasts, given a design matrix.

    Parameters:
    design_matrix (pandas.DataFrame): The design matrix of the experiment.

    Returns:
    dict: A dictionary with keys as contrast names and values as contrast vectors.
    """
    # First, generate canonical contrasts
    contrast_matrix = np.eye(design_matrix.shape[1])
    contrasts = {
        column: contrast_matrix[i]
        for i, column in enumerate(design_matrix.columns)
    }

    # Short dictionary of more relevant contrasts
    contrasts = {
        "+1:image:god-1:image:human": contrasts["image:god"] - contrasts["image:human"],
        "+1:image:inan-1:image:super": contrasts["image:inan"] - contrasts["image:super"]
    }

    return contrasts
def get_contrast_design_matrices(design_matrices, contrast_id):
    """
    Define the contrast array for a given contrast.

    Parameters:
    design_matrices (list of pandas.DataFrame): List of design matrices for each run.
    contrast_id (str): The contrast ID to extract.

    Returns:
    list: List of contrast vectors for each design matrix.
    """
    contrast_val = []

    for design_matrix in design_matrices:
        # Get the dictionary of all contrasts for a given design matrix
        contrasts = make_localizer_contrasts(design_matrix)

        # Append the contrast vector for the given contrast ID
        contrast_val.append(contrasts[contrast_id])

    return contrast_val

def get_mni_fmri_imgs(subject_id, data_root, runs_id):
    """ Return a list of preprocessed MNI-registered fMRI files (Niimg-like object) 
    for fMRI runs.

    Parameters:
    -----------
        subject_id : string
            A string specifying Subject ID.
        data_root : string
            A full path to data in the BIDS format.
        runs_id : list of interger    
            A list of fMRI runs to analyze

    Returns:
    --------
    list of string
        A list of fMRI file names.     

    Examples:
    ---------
    fmri_imgs = utils_fmri_imgs_events.get_fmri_imgs_events(
        subject_id='sub-45TDGV',
        data_root='/project/6019337/databases/religion/bids/',
        runs_id=[1,2,3]
    )

    Note. A Niimg-like object can be one of the following:
        A string or pathlib.Path object with a file path to a Nifti or Analyse image.
        An SpatialImage from nibabel, ie an object exposing get_fdata() method and affine attribute, 
        typically a Nifti1Image from nibabel.
    """
    fmri_imgs = []

    # Get a list of pre-processed FMRI file names  
    for run_id in runs_id:
        fmri_file = os.path.join(
            data_root,
            'preprocessed_23.1.3',
            subject_id,
            'ses-1/func',
            f"{subject_id}_ses-1_task-image_run-{run_id}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
        )
        fmri_imgs.append(fmri_file)

    # Print the list of fMRI files
    print(fmri_imgs)
    
    # Check if all fMRI files exist
    all_exist = all(os.path.exists(f) for f in fmri_imgs)
    
    # Print detailed information about missing fMRI files
    for f in fmri_imgs:
        if not os.path.exists(f):
            print(f"File does not exist: {f}")

    # Print the result
    if all_exist:
        print("All files exist.")
    else:
        sys.exit("Some pre-processed MNI-registered files do not exist.")
       
    return fmri_imgs

def get_events_from_bids(subject_id, data_root, runs_id):
    """ Return a list of events files for fMRI runs.

    Parameters:
    -----------
        subject_id : string
            A string specifying Subject ID.
        data_root : string
            A full path to data in the BIDS format.
        runs_id : list of integer    
            A list of fMRI runs to analyze

    Returns:
    --------
    list of Panda DataFrame
        A list of DataFrames, containing the events.      

    Examples:
    ---------
    event_matrices = utils_fmri_imgs_events.get_events_from_bids(
        subject_id='sub-45TDGV',
        data_root='/projec6019337/databases/religion/bids/',
        runs_id=[1,2,3]
    )
    """
    
    event_files = []
    
    for run_id in runs_id:
        event_file = os.path.join(
            data_root,
            'main',
            subject_id,
            'ses-1/func',
            f"{subject_id}_ses-1_task-image_run-{run_id}_events.tsv"
        )
        event_files.append(event_file)
    
    # Print the list of event files
    print(event_files)
    
    # Check if all event files exist
    all_exist = all(os.path.exists(f) for f in event_files)

    # Print detailed information about missing event files
    for f in event_files:
        if not os.path.exists(f):
            print(f"File does not exist: {f}")
    
    # Print the result
    if all_exist:
        print("All files exist.")
    else:
        sys.exit("Some events files do not exist.")

    # Read each event matrix from a CSV file with tab separation
    event_matrices = [pd.read_csv(df, sep='\t') for df in event_files]
    
    # Apply the function to the 'trial_type' column to set the 'duration' values for each design matrix
    for i, event_matrix in enumerate(event_matrices):
        event_matrix['duration'] = event_matrix['trial_type'].apply(set_event_duration)
    
    # Check the event matrices
    for i, event_matrix in enumerate(event_matrices):
        print(f"Updated Event Matrix for Run {i+1}:")
        print(event_matrix.head(), "\n")

    return event_matrices

def get_mni_fmri_rt(subject_id, data_root, runs_id):
    """ Return the repetition time (RT) of MNI-registered fMRI files.
    Assume that it is the same value for all fMRI runs.

    Parameters:
    -----------
        subject_id : string
            A string specifying Subject ID.
        data_root : string
            A full path to data in the BIDS format.
        runs_id : list of interger    
            A list of fMRI runs to analyze

    Returns:
    --------
    float
        Repetition time, in seconds.
        It is the same value across the fMRI runs.
        Troubles, if not.

    Examples:
    ---------
    fmri_imgs = utils_fmri_imgs_events.get_fmri_imgs_events(
        subject_id='sub-45TDGV',
        data_root='/project/6019337/databases/religion/bids/',
        runs_id=[1,2,3]
    )

    Note. A Niimg-like object can be one of the following:
        A string or pathlib.Path object with a file path to a Nifti or Analyse image.
        An SpatialImage from nibabel, ie an object exposing get_fdata() method and affine attribute, 
        typically a Nifti1Image from nibabel.
    """
    rt_times = []

    # Get a list of pre-processed FMRI file names  
    for run_id in runs_id:
        json_file = os.path.join(
            data_root,
            'preprocessed_23.1.3',
            subject_id,
            'ses-1/func',
            f"{subject_id}_ses-1_task-image_run-{run_id}_space-MNI152NLin2009cAsym_desc-preproc_bold.json"
        )

        # Define the repetition time (TR) of the fMRI data - from corresponding JSON 
        with open(json_file, 'r') as f:
            params = json.load(f)

        # Get the value for the field 'RepetitionTime'
        # and define repetition time (t_r), in seconds
        rt_times.append(params.get('RepetitionTime'))

    if not rt_times:
        raise ValueError("The list of RT times is empty.")
    
    first_value = rt_times[0]
    
    for value in rt_times:
        if value != first_value:
            raise ValueError("Not all RT values in the list are the same.")
    
    return rt_times, first_value

def get_mni_fmri_design_matrices(fmri_imgs, event_matrices, rt_times, hrf_model='glover', drift_model='polynomial', high_pass=0.01, drift_order=3):
    """
    Create design matrices for each fMRI run based on the provided event matrices and repetition times.

    PARAMETERS:
    fmri_imgs: list of str
        List of file paths to fMRI NIFTI images.
    event_matrices: list of pandas.DataFrame
        List of event matrices corresponding to each fMRI run.
    rt_times: list of float
        List of repetition times (TR) for each fMRI run.
    hrf_model: str, optional
        Hemodynamic response function model to use (default is 'glover').
    drift_model: str, optional
        Drift model to use (default is 'polynomial').
    high_pass: float, optional
        High pass filter cutoff in Hz (default is 0.01).
    drift_order: int, optional
        Order of the polynomial drift model (default is 3).

    RETURNS:
    design_matrices: list of pandas.DataFrame
        List of design matrices for each fMRI run.
    """
    # Create a list to store design matrices for each run
    design_matrices = []
    
    for fmri_file, events, t_r in zip(fmri_imgs, event_matrices, rt_times):
        # Load fMRI image
        img = image.load_img(fmri_file)
    
        # The number of frames (or volumes) is the last dimension of the shape
        n_scans = img.shape[-1]
    
        # Compute the frame times
        frame_times = np.arange(n_scans) * t_r
        
        print(f"The repetition time is: {t_r}")
        
        # Create the design matrix
        design_matrix = make_first_level_design_matrix(
            frame_times=frame_times,
            events=events,
            hrf_model=hrf_model,
            drift_model=drift_model,
            high_pass=high_pass,
            drift_order=drift_order,
        )
        
        # Append the design matrix to the list
        design_matrices.append(design_matrix)
    
    return design_matrices
