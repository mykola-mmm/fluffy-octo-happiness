import os
import pandas as pd
import tensorflow as tf
from .rle import rle_to_mask
from pathlib import Path

def classification_data_loader(df_x, df_y, dataset_path, batch_size=32, random_state=42):
    df = pd.concat([df_x, df_y], axis=1)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    while True:
        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i:i+batch_size]
            images = []
            labels = []
            for _, row in batch_df.iterrows():
                image_path = os.path.join(dataset_path, f"{row['ImageId']}")
                image = tf.io.read_file(image_path)
                image = tf.image.decode_jpeg(image, channels=3)
                image = tf.cast(image, tf.float32)
                images.append(image)
                labels.append(row['HasShip'])
            yield tf.stack(images), tf.expand_dims(tf.stack(labels), axis=-1)

def segmentation_data_loader(df_x, df_y, dataset_path, batch_size=32, random_state=42):
    df = pd.concat([df_x, df_y], axis=1)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    while True:
        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i:i+batch_size]
            images = []
            labels = []
            for _, row in batch_df.iterrows():
                image_path = os.path.join(dataset_path, f"{row['ImageId']}")
                image = tf.io.read_file(image_path)
                image = tf.image.decode_jpeg(image, channels=3)
                image = tf.cast(image, tf.float32)
                images.append(image)
                
                # Transform RLE mask to binary mask
                rle_mask = row['AllEncodedPixels']
                if isinstance(rle_mask, str):
                    # Convert RLE to binary mask
                    mask = rle_to_mask(rle_mask, height=768, width=768)  # Adjust dimensions as needed
                    mask = tf.convert_to_tensor(mask, dtype=tf.float32)
                else:
                    # If no mask (empty RLE), create zero mask
                    mask = tf.zeros((768, 768), dtype=tf.float32)
                
                labels.append(mask)
            
            # Stack with proper dimensions
            yield tf.stack(images), tf.stack(labels)[..., tf.newaxis]

def inference_data_loader(images_dir, batch_size=32):
    image_files = list(Path(images_dir).glob('*.jpg'))
    image_files.extend(list(Path(images_dir).glob('*.jpeg')))  # Also include .jpeg files

    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i+batch_size]
        # Convert PosixPath objects to strings and store filenames
        filenames = [file.name for file in batch_files]
        images = [tf.io.read_file(str(file)) for file in batch_files]
        images = [tf.image.decode_jpeg(image, channels=3) for image in images]
        images = [tf.cast(image, tf.float32) for image in images]
        yield tf.stack(images), filenames
