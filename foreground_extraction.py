import numpy as np
import cv2 as cv
import os
import glob
from PIL import Image
from tqdm import tqdm
import argparse


def extract_one(
    uint8_image: np.ndarray, 
    block_size: int, 
    delta: int, 
    kernel_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    # Binarize the image using adaptive thresholding to create a black and white mask.
    binarized = cv.adaptiveThreshold(
        src=uint8_image,
        maxValue=1,
        adaptiveMethod=cv.ADAPTIVE_THRESH_MEAN_C,
        thresholdType=cv.THRESH_BINARY_INV,
        blockSize=block_size,
        C=delta
    )

    # Create a kernel for morphological operations.
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    # Dilate the binarized image to connect broken ridges and fill small holes.
    dilated = cv.dilate(binarized, kernel, iterations=1)

    # Find all connected components (i.e., separate white regions) in the dilated image.
    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(dilated, 8, cv.CV_32S)

    # If there's only the background component, return early.
    if num_labels <= 1:
        return binarized, dilated, dilated, uint8_image
    
    # Find the label of the largest component (ignoring the background at index 0).
    areas = stats[1:, cv.CC_STAT_AREA]
    foreground_label = np.argmax(areas) + 1

    # Create a mask containing only the largest component.
    foreground_mask = np.zeros_like(dilated, dtype=np.uint8)
    foreground_mask[labels == foreground_label] = 1

    # Get the bounding box coordinates of the largest component.
    x = stats[foreground_label, cv.CC_STAT_LEFT]
    y = stats[foreground_label, cv.CC_STAT_TOP]
    w = stats[foreground_label, cv.CC_STAT_WIDTH]
    h = stats[foreground_label, cv.CC_STAT_HEIGHT]
    # Crop the original image to the bounding box of the foreground.
    foreground = uint8_image[y:y+h, x:x+w]

    return binarized, dilated, foreground_mask, foreground


def extract_folder(
    input_folder: str, 
    output_folder: str, 
    block_size: int, 
    delta: int, 
    kernel_size: int
) -> None:
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Supported image extensions
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    
    # Find all image files in the input folder
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(input_folder, ext)))
        image_files.extend(glob.glob(os.path.join(input_folder, ext.upper())))

    print(f"Found {len(image_files)} images in '{os.path.basename(input_folder)}'")
    
    # Process each image
    for image_path in tqdm(image_files, desc=f"Processing images in '{os.path.basename(input_folder)}'"):
        try:
            # Load image
            pil = Image.open(image_path).convert('L') # Convert to grayscale
            original = np.array(pil).astype(np.uint8)

            # Extract foreground using the extract_one() function
            _, _, _, foreground = extract_one(original, block_size, delta, kernel_size)
            
            # Save the foreground image to the output folder
            output_path = os.path.join(output_folder, f"{os.path.basename(image_path)}")
            Image.fromarray(foreground).save(output_path)
            
        except Exception as e:
            # Use tqdm.write to print messages without breaking the progress bar
            tqdm.write(f"Error processing {os.path.basename(image_path)}: {str(e)}")
            continue
    
    print(f"Processing complete! Foreground images saved to {output_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fingerprint Foreground Extraction')
    parser.add_argument('-i', '--input_folder', type=str, required=True,
                        help='Input folder containing fingerprint images')
    parser.add_argument('-o', '--output_folder', type=str, default='./foregrounds',
                        help='Output folder for foreground images (default: ./foregrounds)')
    parser.add_argument('-b', '--block_size', type=int, default=5,
                        help='Block size for adaptive thresholding (default: 5)')
    parser.add_argument('-d', '--delta', type=int, default=15,
                        help='Delta value for adaptive thresholding (default: 15)')
    parser.add_argument('-k', '--kernel_size', type=int, default=5,
                        help='Kernel size for morphological operations (default: 5)')
    
    args = parser.parse_args()
    extract_folder(args.input_folder, args.output_folder, args.block_size, args.delta, args.kernel_size)
