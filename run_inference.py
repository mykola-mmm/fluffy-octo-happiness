import logging	
import tensorflow as tf
import pandas as pd
from pathlib import Path
from src.args.run_inference_args import process_inference_args
from src.utils.data_loader import inference_data_loader
from matplotlib import pyplot as plt
# from src.models.segmentation_model import SegmentationModel


logging.basicConfig(
    level=logging.INFO,
    format='%(name)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)

def main():
    args = process_inference_args()
    logger.debug(f"Args: {args}")

    # classification_model = tf.keras.models.load_model(args.classification_model_path)
    # classification_model = tf.keras.models.load_model(args.classification_model_path)
    segmentation_model = tf.keras.models.load_model(args.segmentation_model_path, compile=False)
    logger.info(f"Segmentation model summary: {segmentation_model.summary()}")
    
    # segmentation_model = tf.keras.models.load_model(args.segmentation_model_path)

    # Get all jpg files from the directory
    image_files = list(Path(args.images_dir).glob('*.jpg'))
    image_files.extend(list(Path(args.images_dir).glob('*.jpeg')))  # Also include .jpeg files
    
    # Create DataFrame with image paths
    df = pd.DataFrame({
        'file_name': [f.name for f in image_files]
    })

    # Save DataFrame to CSV
    df.to_csv(args.output_csv_path, index=False)
    logger.info(f"Saved results to {args.output_csv_path}")

    data_loader = inference_data_loader(args.images_dir, batch_size=args.batch_size)

    inference_loader = tf.data.Dataset.from_generator(
        lambda: data_loader,
        output_signature=tf.TensorSpec(shape=(None, 768, 768, 3), dtype=tf.float32)
    )

    for i, batch in enumerate(inference_loader.take(10)):
        # predictions = []
        pred = segmentation_model(batch)
        # pred = segmentation_model.predict(batch, verbose=0)
        logger.info(f"pred.shape: {pred.shape}")
        logger.info(f"batch.shape: {batch.shape}")
        # predictions.append(pred)
        
        # Iterate through each image in the batch
        for j in range(batch.shape[0]):
            plt.figure(figsize=(12, 6))
            
            # Display original image
            plt.subplot(1, 2, 1)
            plt.imshow(batch[j]/255)
            plt.title('Original Image')
            plt.axis('off')
            
            # Display prediction mask
            plt.subplot(1, 2, 2)
            # plt.imshow(pred[j], cmap='gray')  # Assuming single channel output
            plt.imshow(pred[j, :, :, 0], cmap='gray')  # Access the first channel
            plt.title('Prediction Mask')
            plt.axis('off')
            
            plt.show()

if __name__ == "__main__":
    logger.info("Starting inference")

    # Set mixed precision policy
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        main()

    logger.info("Completed inference")
