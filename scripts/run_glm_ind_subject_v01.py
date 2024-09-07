# Create a dictionary to store design matrices
design_matrices = {}

# Loop over fMRI images and their corresponding event matrices
for fmri_img, events in zip(fmri_imgs, event_matrices):
    # Load the fMRI image
    img = load_img(fmri_img)
    
    # Create the design matrix
    design_matrix = make_first_level_design_matrix(
        frame_times=frame_times,
        events=events,
        hrf_model='glover',
        drift_model='polynomial',
        high_pass=0.01,
        drift_order=3,
    )
    
    # Store the design matrix in the dictionary with the fMRI image path as the key
    design_matrices[fmri_img] = design_matrix

# Print the design matrices for verification
for fmri_img, design_matrix in design_matrices.items():
    print(f"Design Matrix for {fmri_img}:")
    print(design_matrix.head())
    print()