import sys
import os
import pandas as pd
import random
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import mixed_precision
from sklearn.model_selection import train_test_split

# from src.data_loader import load_data
# from src.model.segmentation import UNet
from src.model.classification import BinaryClassificationCNN
from src.data_loader import classification_data_loader, classification_validation_data_loader
# from src.model.bigEarthNet import model as bigEarthNetModel 
# from src.trainer import Trainer
from src.utils import setup_logging, get_log_level, parse_arguments
from src.csv_preprocessor import preprocess_csv, get_ships_df, get_no_ships_df, save_df_to_csv


def main():
    # Parse command-line arguments
    args = parse_arguments()

    # Set mixed precision policy
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)

    # Set the log level as an environment variable
    os.environ['LOG_LEVEL'] = args.log_level

    # Set up logging
    logger = setup_logging()
    logger.info("Starting the application...")

    # Use the parsed arguments
    logger.info(f"Data path: {args.dataset_path}")
    logger.info(f"Log level: {get_log_level()}")
    logger.info(f"Task: {args.task}")

    CLASSIFICATION_BATCH_SIZE = 16

    # Add your main application logic here
    if args.task == "train_classification":
        logger.info("Starting train_classification task...")
        model = BinaryClassificationCNN()
        df_ships = pd.read_csv(os.path.join(args.processed_csv_dir, "df_ships.csv"))
        df_no_ships = pd.read_csv(os.path.join(args.processed_csv_dir, "df_no_ships.csv"))

        df = pd.concat([df_ships, df_no_ships], ignore_index=True)
        x_train, x_val, y_train, y_val = train_test_split(df['ImageId'], df['HasShip'], test_size=0.2, stratify=df['HasShip'], random_state=42)

        _train_loader = classification_data_loader(x_train, y_train, args.dataset_path, batch_size=CLASSIFICATION_BATCH_SIZE)
        _validation_loader = classification_data_loader(x_val, y_val, args.dataset_path, batch_size=CLASSIFICATION_BATCH_SIZE)

        # train_loader = _train_loader
        # validation_loader = _validation_loader

        train_loader = tf.data.Dataset.from_generator(
            lambda: _train_loader,
            output_types=(tf.float16, tf.int32),
            output_shapes=((None, 768, 768, 3), (None,))
        )

        validation_loader = tf.data.Dataset.from_generator(
            lambda: _validation_loader,
            output_types=(tf.float16, tf.int32),
            output_shapes=((None, 768, 768, 3), (None,))
        )    
        
        # images, labels = next(train_loader)
        # print(images.shape, labels.shape)

        # Generate class weights because of imbalanced dataset
        total_samples = len(y_train)
        num_zero = (y_train == 0).sum()
        num_one = (y_train == 1).sum()
        weight_zero = (1 / num_zero) * (total_samples / 2.0)
        weight_one = (1 / num_one) * (total_samples / 2.0)

        # Normalize weights to sum to 1
        total_weight = weight_zero + weight_one
        weight_zero /= total_weight
        weight_one /= total_weight

        logger.info(f"Class weights - Zero: {weight_zero:.4f}, One: {weight_one:.4f}")

        model.compile_model(learning_rate=0.0001, weight_zero=weight_zero, weight_one=weight_one)
        model.summary()
        model.train(train_loader,
                    validation_loader,
                    epochs=25,
                    train_df_len=len(x_train),
                    validation_df_len=len(x_val),
                    batch_size=CLASSIFICATION_BATCH_SIZE,
                    save_path=args.classification_model_path,
                    weight_zero=weight_zero,
                    weight_one=weight_one)

    elif args.task == "test_classification":
        logger.info("Starting test_classification task...")
        model = BinaryClassificationCNN()
        model.load_weights(args.classification_model_path)
        model.summary()

        # Load random images and generate predictions
        image_files = [f for f in os.listdir(args.dataset_path) if f.endswith('.jpg') or f.endswith('.png')]
        selected_images = random.sample(image_files, min(args.num_images, len(image_files)))

        for image_file in selected_images:
            image_path = os.path.join(args.dataset_path, image_file)
            
            # Read and preprocess the image using TensorFlow operations
            img = tf.io.read_file(image_path)
            img = tf.image.decode_image(img, channels=3)
            img_human_readable = tf.identity(img)
            img = tf.keras.applications.vgg19.preprocess_input(img)
            # img = tf.image.resize(img, (224, 224))
            # img = tf.cast(img, tf.float32) / 255.0
            img_array = tf.expand_dims(img, axis=0)

            prediction = model.predict(img_array)
            print(f"prediction.shape: {prediction.shape}")
            predicted_class = "Ship" if prediction[0][0] > 0.5 else "No Ship"

            plt.figure(figsize=(8, 6))
            plt.imshow(img_human_readable)
            plt.title(f"Prediction: {predicted_class} (Confidence: {prediction[0][0]:.4f})")
            plt.axis('off')
            plt.show()

            logger.info(f"Image: {image_file}, Prediction: {predicted_class}, Prediction: {prediction[0][0]:.2f}")

    elif args.task == "train_segmentation":
        logger.info("Starting train_segmentation task...")

        
    elif args.task == "infer":
        logger.info("Starting inference task...")

        
    elif args.task == "preprocess":
        logger.info("Starting data preprocessing task...")
        logger.info(f"CSV file: {args.raw_csv_file}")
        
        df = preprocess_csv(args.raw_csv_file)
        df_ships = get_ships_df(df)
        df_no_ships = get_no_ships_df(df)

        save_df_to_csv(df_ships, os.path.join(args.processed_csv_dir, "df_ships.csv"))
        save_df_to_csv(df_no_ships, os.path.join(args.processed_csv_dir, "df_no_ships.csv"))

if __name__ == "__main__":
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        main()
