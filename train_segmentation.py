import os
import logging
import pandas as pd
import tensorflow as tf
from src.models.segmentation_model import SegmentationModel
from sklearn.model_selection import train_test_split
from src.utils.data_loader import segmentation_data_loader
from src.args.train_segmentation_args import process_csv_args

import matplotlib.pyplot as plt


logging.basicConfig(
    level=logging.DEBUG,
    format='%(name)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def main():
    args = process_csv_args()
    logger.debug(f"Args: {args}")

    try:
        df = pd.read_csv(args.csv_file_path)
        logger.info(f"Successfully read CSV file: {args.csv_file_path}")
    except FileNotFoundError:
        logger.error(f"File not found: {args.csv_file_path}")
        return

    x_train, x_val, y_train, y_val = train_test_split(df['ImageId'], df['AllEncodedPixels'], test_size=0.2, stratify=df['ShipCount'], random_state=args.rand_seed)
    train_data_loader = segmentation_data_loader(x_train, y_train, args.dataset_path, batch_size=args.batch_size, random_state=args.rand_seed)
    validation_data_loader = segmentation_data_loader(x_val, y_val, args.dataset_path, batch_size=args.batch_size, random_state=args.rand_seed)

    train_loader = tf.data.Dataset.from_generator(
        lambda: train_data_loader,
        output_signature=(
            tf.TensorSpec(shape=(None, 768, 768, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 768, 768, 1), dtype=tf.int16)
        )
    )

    validation_loader = tf.data.Dataset.from_generator(
        lambda: validation_data_loader,
        output_signature=(
            tf.TensorSpec(shape=(None, 768, 768, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 768, 768, 1), dtype=tf.int16)
        )
    )

    # # Visualize first batch
    # for images, masks in train_loader.take(1):
    #     # Take the first batch and convert to numpy for visualization
    #     logger.debug(f"Images shape: {images.shape}, Masks shape: {masks.shape}")
    #     images = images.numpy()
    #     masks = masks.numpy()
        
    #     for i in range(images.shape[0]):
    #         # Create a figure with 2 rows for each image
    #         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 12))
            
    #         # Display original image
    #         ax1.imshow(images[i]/255)
    #         ax1.set_title(f'Image {i+1}')
    #         ax1.axis('off')
            
    #         # Display mask
    #         ax2.imshow(masks[i, :, :, 0], cmap='gray')
    #         ax2.set_title(f'Mask {i+1}')
    #         ax2.axis('off')
            
    #         plt.tight_layout()
    #         plt.show()

    train_steps_per_epoch = len(x_train) // args.batch_size
    val_steps_per_epoch = len(x_val) // args.batch_size

    model = SegmentationModel(
        dropout_rate=args.dropout_rate,
        pretrained=args.pretrained
    )
    # model.summary()
    # model.visualize_model()
    model.set_backbone_trainable(trainable=True)

    model.compile_model(steps_per_epoch=train_steps_per_epoch, learning_rate = args.learning_rate, decay_rate=args.decay_rate)
    model.train(
        train_data_loader=train_loader,
        val_data_loader=validation_loader,
        epochs=args.epochs,
        train_steps_per_epoch=train_steps_per_epoch,
        val_steps_per_epoch=val_steps_per_epoch,
        save_path=args.save_path,
        logs_path=args.logs_path,
        early_stopping_patience=args.early_stopping_patience
    )

    # Save the trained model
    save_path = os.path.join(args.save_path, 'final_model.keras')
    save_path_weights = os.path.join(args.save_path, 'final_model.weights.h5')
    model.save(save_path)
    model.save_weights(save_path_weights)
    logger.info(f"Model saved to: {save_path}")
    logger.info(f"Model summary: {model.summary()}")

    # Visualize first batch from validation set
    for images, masks in validation_loader.take(10):
        # Get model predictions for the batch
        predictions = model.predict(images)
        
        # Convert tensors to numpy arrays
        logger.debug(f"Images shape: {images.shape}, Masks shape: {masks.shape}")
        images = images.numpy()
        masks = masks.numpy()
        
        for i in range(images.shape[0]):
            # Create a figure with 3 columns for each image
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
            
            # Display original image
            ax1.imshow(images[i]/255)
            ax1.set_title('Original Image')
            ax1.axis('off')
            
            # Display ground truth mask
            ax2.imshow(masks[i, :, :, 0], cmap='gray')
            ax2.set_title('Ground Truth')
            ax2.axis('off')
            
            # Display model prediction
            ax3.imshow(predictions[i, :, :, 0], cmap='gray')
            ax3.set_title('Prediction')
            ax3.axis('off')
            
            plt.tight_layout()
            plt.show()

    # try:
    #     validation_data_loader_test = segmentation_data_loader(x_val, y_val, args.dataset_path, batch_size=args.batch_size, random_state=args.rand_seed)
    #     validation_loader_test = tf.data.Dataset.from_generator(
    #         lambda: validation_data_loader_test,
    #         output_signature=(
    #             tf.TensorSpec(shape=(None, 768, 768, 3), dtype=tf.float32),
    #             tf.TensorSpec(shape=(None, 768, 768, 1), dtype=tf.int16)
    #         )
    #     )
    #     model_test = tf.keras.models.load_model(save_path)
    #     logger.info(f"Successfully loaded model from: {save_path}")
    #         # Visualize first batch from validation set
    #     for images, masks in validation_loader_test.take(1):
    #         # Get model predictions for the batch
    #         predictions = model_test.predict(images)
            
    #         # Convert tensors to numpy arrays
    #         logger.debug(f"Images shape: {images.shape}, Masks shape: {masks.shape}")
    #         images = images.numpy()
    #         masks = masks.numpy()
            
    #         for i in range(images.shape[0]):
    #             # Create a figure with 3 columns for each image
    #             fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
                
    #             # Display original image
    #             ax1.imshow(images[i]/255)
    #             ax1.set_title('Original Image')
    #             ax1.axis('off')
                
    #             # Display ground truth mask
    #             ax2.imshow(masks[i, :, :, 0], cmap='gray')
    #             ax2.set_title('Ground Truth')
    #             ax2.axis('off')
                
    #             # Display model prediction
    #             ax3.imshow(predictions[i, :, :, 0], cmap='gray')
    #             ax3.set_title('Prediction')
    #             ax3.axis('off')
                
    #             plt.tight_layout()
    #             plt.show()
    # except Exception as e:
    #     logger.error(f"Error loading model from {save_path}: {str(e)}")
    #     raise


    

#     model = ClassificationModel(
#         dropout_rate=args.dropout_rate,
#         pretrained=args.pretrained
#     )

#     # Calculate steps per epoch
#     train_steps_per_epoch = len(x_train) // args.batch_size
#     val_steps_per_epoch = len(x_val) // args.batch_size

#     # Transfer Learning phase
#     model.set_backbone_trainable(trainable=True)
#     model.compile_model(stage="tl", steps_per_epoch=train_steps_per_epoch, learning_rate = args.tl_learning_rate, decay_rate=args.decay_rate)
#     logger.debug(f"Model summary: {model.summary()}")

#     model.train(
#         stage="tl",
#         train_data_loader=train_loader,
#         val_data_loader=validation_loader,
#         epochs=args.tl_epochs,
#         train_steps_per_epoch=train_steps_per_epoch,
#         val_steps_per_epoch=val_steps_per_epoch,
#         save_path=args.save_path,
#         logs_path=args.logs_path,
#         early_stopping_patience=args.early_stopping_patience
#     )

#     logger.info(f"Transfer Learning Training Completed")

#     model.visualize_history('tl')
#     model.run_inference(validation_loader)



if __name__ == "__main__":
    logger.info("Starting train_classification")

    # Set mixed precision policy
    # policy = tf.keras.mixed_precision.Policy('mixed_float16')
    # tf.keras.mixed_precision.set_global_policy(policy)

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        main()

    logger.info("Completed train_classification")
