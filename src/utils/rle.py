import numpy as np
import pandas as pd
from scipy import ndimage

def rle_to_mask(rle_string, height=768, width=768):
    """Convert RLE string to binary mask and apply transformations:
    1. Rotate 90 degrees counterclockwise
    2. Mirror along x-axis
    """
    if pd.isna(rle_string):
        return np.zeros((width, height))  # Note: dimensions swapped due to rotation
    
    mask = np.zeros(height * width)
    array = np.asarray([int(x) for x in rle_string.split()])
    starts = array[0::2]
    lengths = array[1::2]

    for start, length in zip(starts, lengths):
        mask[start-1:start-1+length] = 1
    
    # Reshape, rotate 90° counterclockwise, and mirror along x-axis
    mask = mask.reshape(height, width)
    mask = np.rot90(mask)  # Rotate 90° counterclockwise
    mask = np.flipud(mask)  # Mirror along x-axis
    
    return mask

def rle_decode(mask):
    """Convert binary mask to RLE string."""
    # Return NaN if mask is empty or contains no positive values
    if mask.size == 0 or not np.any(mask):
        return np.nan
        
    # Apply inverse transformations
    mask = np.flipud(mask)  # Inverse of mirror along x-axis
    mask = np.rot90(mask, k=-1)  # Inverse of 90° counterclockwise rotation
    
    pixels = mask.flatten()
    starts = []
    lengths = []
    
    for i in range(len(pixels)):
        if pixels[i] == 1:
            starts.append(i)
            while i < len(pixels) and pixels[i] == 1:
                i += 1
            lengths.append(i - starts[-1])
    return ' '.join(f"{start} {length}" for start, length in zip(starts, lengths))

def split_mask(mask):
    """Split a binary mask into separate masks for each connected component.
    
    Args:
        mask (np.ndarray): Binary mask containing multiple objects
        
    Returns:
        list: List of binary masks, one for each detected object
    """
    # Ensure mask is binary and of correct dtype
    mask = (mask > 0).astype(np.int32)
    
    # Label connected components in the mask
    labeled_array, num_features = ndimage.label(mask)
    
    # Create separate masks for each labeled component
    individual_masks = []
    for label in range(1, num_features + 1):
        individual_mask = (labeled_array == label).astype(np.uint8)
        individual_masks.append(individual_mask)
    
    return individual_masks

