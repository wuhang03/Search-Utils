import os
import numpy as np
import cv2

pred_dir = './Test_pred'
gt_dir = './Test_GT'
artifact_dir = './Test_artifact'

# Create the artifact directory if it doesn't exist
os.makedirs(artifact_dir, exist_ok=True)

# Get the list of files in the pred directory
pred_files = os.listdir(pred_dir)

for file in pred_files:
    # Get the file index
    file_index = file.split('.')[0]

    # Read the pred and gt files
    pred_file = os.path.join(pred_dir, file)
    gt_file = os.path.join(gt_dir, file)
    pred_img = cv2.imread(pred_file, cv2.IMREAD_ANYDEPTH)
    gt_img = cv2.imread(gt_file, cv2.IMREAD_ANYDEPTH)

    # Subtract the images
    artifact_img = np.subtract(gt_img, pred_img)

    # Save the artifact image
    artifact_file = os.path.join(artifact_dir, f'{file_index}.hdr')
    cv2.imwrite(artifact_file, artifact_img)

    # Print the artifact file path
    print(f'Artifact file created: {artifact_file}')

print('Artifact files created successfully.')
