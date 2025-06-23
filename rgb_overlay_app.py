import cv2
import numpy as np
import os

# Paths
input_folder = "task_1_output/input"
output_folder = "task_1_output/output"

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Get all thermal images in the input folder
thermal_images = [f for f in os.listdir(input_folder) if f.endswith('_T.JPG')]

# Function to align and overlay thermal on RGB
def align_and_overlay(thermal_path, rgb_path, output_path):
    # Read images
    rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    thermal = cv2.imread(thermal_path, cv2.IMREAD_COLOR)

    # Resize thermal image to match RGB size before alignment
    thermal = cv2.resize(thermal, (rgb.shape[1], rgb.shape[0]))


    # Convert to grayscale for keypoint detection
    gray_thermal = cv2.cvtColor(thermal, cv2.COLOR_BGR2GRAY)
    gray_rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

    # Use ORB detector
    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(gray_thermal, None)
    kp2, des2 = orb.detectAndCompute(gray_rgb, None)

    # Match features using BFMatcher
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Get matched keypoints
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches[:50]]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches[:50]]).reshape(-1, 1, 2)

    # Estimate homography matrix
    matrix, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Warp thermal image to match RGB image
    aligned_thermal = cv2.warpPerspective(thermal, matrix, (rgb.shape[1], rgb.shape[0]))

    # Blend the aligned thermal image onto the RGB image
    overlay = cv2.addWeighted(rgb, 0.6, aligned_thermal, 0.4, 0)

    # Save result
    cv2.imwrite(output_path, overlay)

# Process all pairs
for thermal_file in thermal_images:
    # Get shared ID
    image_id = thermal_file.replace('_T.JPG', '')
    rgb_file = image_id + '_Z.JPG'

    thermal_path = os.path.join(input_folder, thermal_file)
    rgb_path = os.path.join(input_folder, rgb_file)
    output_path = os.path.join(output_folder, image_id + '_AT.JPG')

    # Check if corresponding RGB exists
    if os.path.exists(rgb_path):
        print(f"[INFO] Processing pair: {thermal_file} + {rgb_file}")
        align_and_overlay(thermal_path, rgb_path, output_path)
    else:
        print(f"[WARNING] Skipping {thermal_file}, RGB pair not found.")
