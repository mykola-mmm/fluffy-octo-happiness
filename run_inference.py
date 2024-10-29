import logging	
import tensorflow as tf
import pandas as pd
import numpy as np
import csv
from pathlib import Path
from src.args.run_inference_args import process_inference_args
from src.utils.data_loader import inference_data_loader
from matplotlib import pyplot as plt
from src.models.segmentation_model import SegmentationModel
from src.utils.dice_loss import DiceLoss, CombinedLoss
from src.utils.iou import IoU
from src.utils.rle import rle_decode, rle_to_mask, split_mask


logging.basicConfig(
    level=logging.DEBUG,
    format='%(name)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)

def main():
    args = process_inference_args()
    logger.debug(f"Args: {args}")

    # Load the model from .h5 file
    logger.info(f"Loading model from {args.segmentation_model_path}")
    custom_objects = {
        'SegmentationModel': SegmentationModel,
        'DiceLoss': DiceLoss,
        'CombinedLoss': CombinedLoss,
        'IoU': IoU
    }

    segmentation_model = tf.keras.models.load_model(args.segmentation_model_path, custom_objects=custom_objects)

    logger.info("Model loaded successfully")

    # # Get all jpg files from the directory
    # image_files = list(Path(args.images_dir).glob('*.jpg'))
    # image_files.extend(list(Path(args.images_dir).glob('*.jpeg')))  # Also include .jpeg files
    
    # # Create DataFrame with image paths
    # df = pd.DataFrame({
    #     'ImageId': [f.name for f in image_files]
    # })

    # # Save DataFrame to CSV
    # df.head(100)
    # df.to_csv(args.output_csv_path, index=False)
    # logger.info(f"Saved results to {args.output_csv_path}")

    data_loader = inference_data_loader(args.images_dir, batch_size=args.batch_size)

    inference_loader = tf.data.Dataset.from_generator(
        lambda: data_loader,
        output_signature=(
            tf.TensorSpec(shape=(None, 768, 768, 3), dtype=tf.float32),  # images
            tf.TensorSpec(shape=(None,), dtype=tf.string)  # filenames
        )
    )

    # Create DataFrame with image paths and empty EncodedPixels column
    df = pd.DataFrame(columns=['ImageId', 'EncodedPixels'])
    
    # Iterate through first 100 batches in the inference loader
    for batch_num, (batch_images, batch_filenames) in enumerate(inference_loader):
        # if batch_num >= 10:
        #     break
            
        # logger.info(f"Processing batch {batch_num}")
        # logger.info(f"batch_images.shape: {batch_images.shape}")
        # logger.info(f"batch_filenames.shape: {batch_filenames.shape}")
            
        # Use model.predict for inference
        predictions = segmentation_model.predict(batch_images, verbose=0)

        # logger.info(f"predictions.shape: {predictions.shape}")
        # logger.info(f"batch_filenames.shape: {batch_filenames.shape}")
        
        # Process each image in the batch
        for img_pred, filename in zip(predictions, batch_filenames):
            image_id = filename.numpy().decode('utf-8')  # Extract image_id once
            
            # logger.info(f"Processing image: {image_id}")

            if img_pred.sum() > 0:
                masks = split_mask(img_pred)
                for mask in masks:
                    decoded_mask = rle_decode(mask)
                    logger.info(f"type(decoded_mask): {type(decoded_mask)}")
                    decoded_mask = str(decoded_mask).replace('\n', ' ')
                    # Replace any newlines with spaces to ensure single-line output
                    new_row = pd.DataFrame({
                        'ImageId': [image_id],
                        'EncodedPixels': [decoded_mask]
                    }, columns=['ImageId', 'EncodedPixels'])
                    df = pd.concat([df, new_row], ignore_index=True)
            else:
                new_row = pd.DataFrame({
                    'ImageId': [image_id],
                    'EncodedPixels': [np.nan]
                }, columns=['ImageId', 'EncodedPixels'])  # Explicitly specify column order
                df = pd.concat([df, new_row], ignore_index=True)

    # Clean and format the DataFrame before saving
    def clean_encoded_pixels(x):
        if pd.isna(x):
            return ''
        return ' '.join(str(x).split())  # This will normalize spaces and remove any hidden characters

    df['EncodedPixels'] = df['EncodedPixels'].apply(clean_encoded_pixels)

    # Write CSV manually to ensure proper formatting
    with open(args.output_csv_path, 'w', newline='') as f:
        f.write('ImageId,EncodedPixels\n')  # Write header
        for _, row in df.iterrows():
            image_id = row['ImageId']
            encoded_pixels = row['EncodedPixels']
            # Properly quote and format each line
            f.write(f'"{image_id}","{encoded_pixels}"\n')

    logger.info(f"Saved results to {args.output_csv_path}")

    # # Add validation visualization
    # logger.info("Starting validation visualization")
    # for idx, row in df.iterrows():
    #     if pd.notna(row['EncodedPixels']) and row['EncodedPixels']:  # Check if mask exists
    #         # Load and preprocess image
    #         img_path = Path(args.images_dir) / row['ImageId']
    #         image = tf.keras.preprocessing.image.load_img(img_path)
    #         image = tf.keras.preprocessing.image.img_to_array(image)
            
    #         # Decode RLE to get mask
    #         mask = rle_to_mask(row['EncodedPixels'])
            
    #         # Create visualization
    #         plt.figure(figsize=(12, 6))
            
    #         # Original image
    #         plt.subplot(1, 2, 1)
    #         plt.imshow(image.astype('uint8'))
    #         plt.title('Original Image')
    #         plt.axis('off')
            
    #         # Mask overlay
    #         plt.subplot(1, 2, 2)
    #         plt.imshow(mask, cmap='gray')  # Overlay mask with transparency
    #         plt.title('Image with Mask Overlay')
    #         plt.axis('off')
            
    #         plt.show()
            
    #         # Optional: break after a few images to avoid showing too many
    #         if idx >= 50:  # Show first 5 images with masks
    #             break
    
    # for i, batch in enumerate(inference_loader.take(1)):
    #     # Use model.predict instead of direct call for inference
    #     pred = segmentation_model.predict(batch, verbose=0)
        
    #     # Apply threshold to predictions
    #     # pred = tf.where(pred > 0.5, 1.0, 0.0)
        
    #     logger.info(f"pred.shape: {pred.shape}")
    #     logger.info(f"batch.shape: {batch.shape}")
        
    #     # Iterate through each image in the batch
    #     for j in range(batch.shape[0]):
    #         plt.figure(figsize=(12, 6))
            
    #         # Display original image
    #         plt.subplot(1, 2, 1)
    #         plt.imshow(batch[j]/255)
    #         plt.title('Original Image')
    #         plt.axis('off')
            
    #         # Display prediction mask
    #         plt.subplot(1, 2, 2)
    #         # plt.imshow(pred[j], cmap='gray')  # Assuming single channel output
    #         plt.imshow(pred[j, :, :, 0], cmap='gray')  # Access the first channel
    #         plt.title('Prediction Mask')
    #         plt.axis('off')
            
    #         plt.show()

    

if __name__ == "__main__":
    logger.info("Starting inference")

    # Set mixed precision policy
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        main()

    logger.info("Completed inference")
