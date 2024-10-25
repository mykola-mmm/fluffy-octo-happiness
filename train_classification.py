import os
import logging
import pandas as pd
import tensorflow as tf
from src.models.classification_model import ClassificationModel
from sklearn.model_selection import train_test_split
from src.utils.data_loader import classification_data_loader
from src.args.train_classification_args import process_csv_args

import matplotlib.pyplot as plt


logging.basicConfig(
    level=logging.INFO,
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

    x_train, x_val, y_train, y_val = train_test_split(df['ImageId'], df['HasShip'], test_size=0.2, stratify=df['ShipCount'], random_state=args.rand_seed)
    train_data_loader = classification_data_loader(x_train, y_train, args.dataset_path, batch_size=args.batch_size, random_state=args.rand_seed)
    validation_data_loader = classification_data_loader(x_val, y_val, args.dataset_path, batch_size=args.batch_size, random_state=args.rand_seed)

    train_loader = tf.data.Dataset.from_generator(
        lambda: train_data_loader,
        output_signature=(
            tf.TensorSpec(shape=(None, 768, 768, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 1), dtype=tf.int32)
        )
    )

    validation_loader = tf.data.Dataset.from_generator(
        lambda: validation_data_loader,
        output_signature=(
            tf.TensorSpec(shape=(None, 768, 768, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 1), dtype=tf.int32)
        )
    )

    model = ClassificationModel(
        dropout_rate=args.dropout_rate,
        pretrained=args.pretrained,
        l1=args.l1,
        l2=args.l2
    )

    # Calculate steps per epoch
    train_steps_per_epoch = len(x_train) // args.batch_size
    val_steps_per_epoch = len(x_val) // args.batch_size

    # Transfer Learning phase
    model.set_backbone_trainable(trainable=False)
    model.compile_model(stage="tl", steps_per_epoch=train_steps_per_epoch, learning_rate = args.tl_learning_rate, decay_rate=args.tl_decay_rate)
    logger.debug(f"Model summary: {model.summary()}")

    model.train(
        stage="tl",
        train_data_loader=train_loader,
        val_data_loader=validation_loader,
        epochs=args.tl_epochs,
        train_steps_per_epoch=train_steps_per_epoch,
        val_steps_per_epoch=val_steps_per_epoch,
        save_path=args.save_path,
        logs_path=args.logs_path,
        early_stopping_patience=args.early_stopping_patience
    )

    logger.info(f"Transfer Learning Training Completed")

    model.visualize_history('tl')

    # Fine-tuning phase
    model.set_backbone_trainable(trainable=True)  # Unfreeze the backbone
    model.compile_model(stage="ft", learning_rate = args.ft_learning_rate, steps_per_epoch=train_steps_per_epoch, warmup_epochs=args.ft_warmup_epochs, min_learning_rate=args.ft_min_learning_rate, decay_rate=args.ft_decay_rate)

    model.train(
        stage="ft",
        train_data_loader=train_loader,
        val_data_loader=validation_loader,
        epochs=args.ft_epochs,
        train_steps_per_epoch=train_steps_per_epoch,
        val_steps_per_epoch=val_steps_per_epoch,
        save_path=args.save_path,
        logs_path=args.logs_path,
        early_stopping_patience=args.early_stopping_patience
    )

    # model.save_best_model(args.save_path)
    model.visualize_history('ft')
    model.run_inference(validation_loader)



if __name__ == "__main__":
    logger.info("Starting train_classification")

    # Set mixed precision policy
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        main()

    logger.info("Completed train_classification")
