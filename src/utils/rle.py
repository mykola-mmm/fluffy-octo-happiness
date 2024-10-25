import numpy as np
import pandas as pd

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
