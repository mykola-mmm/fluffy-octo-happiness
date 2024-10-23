import logging
import pandas as pd
import tensorflow as tf
from src.models.efficientnet import EfficientNet
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

    x_train, x_val, y_train, y_val = train_test_split(df['ImageId'], df['HasShip'], test_size=0.2, stratify=df['HasShip'], random_state=args.rand_seed)
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

    model = EfficientNet()
    # logger.debug(f"Model summary: {model.summary()}")
    x, y = next(iter(train_loader))
    pred = model(x)[0]
    logger.debug(f"pred: {pred.shape}")
    logger.debug(f"x - {x[0]}")


    logger.debug(f"pred - {pred}")
    logger.debug(f"pred - {pred.dtype}")
    plt.imshow(pred/255)
    plt.show()

if __name__ == "__main__":
    logger.info("Starting train_classification")

    # Set mixed precision policy
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        main()

    logger.info("Completed train_classification")
