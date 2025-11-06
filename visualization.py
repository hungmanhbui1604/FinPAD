import matplotlib.pyplot as plt
import numpy as np


def visualize_foreground_extraction(
    original_image: np.ndarray,
    binarized: np.ndarray,
    dilated: np.ndarray,
    foreground_mask: np.ndarray,
    foreground_crop: np.ndarray
):

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Foreground Extraction Steps', fontsize=16)

    # 1. Display original grayscale image
    axes[0, 0].imshow(original_image, cmap='gray')
    axes[0, 0].set_title('1. Original Grayscale')
    axes[0, 0].axis('off')

    # 2. Display binarized image
    axes[0, 1].imshow(binarized, cmap='gray')
    axes[0, 1].set_title('2. Binarized (Adaptive Threshold)')
    axes[0, 1].axis('off')

    # 3. Display dilated image
    axes[0, 2].imshow(dilated, cmap='gray')
    axes[0, 2].set_title('3. Dilated')
    axes[0, 2].axis('off')

    # 4. Display foreground mask
    axes[1, 0].imshow(foreground_mask, cmap='gray')
    axes[1, 0].set_title('4. Foreground Mask')
    axes[1, 0].axis('off')

    # 5. Display foreground
    axes[1, 1].imshow(foreground_crop, cmap='gray')
    axes[1, 1].set_title('5. Foreground')
    axes[1, 1].axis('off')

    # Turn off the last unused subplot
    axes[1, 2].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()