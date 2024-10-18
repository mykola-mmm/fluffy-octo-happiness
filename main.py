import sys
import os
import pandas as pd

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

    # Set the log level as an environment variable
    os.environ['LOG_LEVEL'] = args.log_level

    # Set up logging
    logger = setup_logging()
    logger.info("Starting the application...")

    # Use the parsed arguments
    logger.info(f"Data path: {args.dataset_path}")
    logger.info(f"Log level: {get_log_level()}")
    logger.info(f"Task: {args.task}")

    # Add your main application logic here
    if args.task == "train_classification":
        logger.info("Starting train_classification task...")
        model = BinaryClassificationCNN()
        df_ships = pd.read_csv(os.path.join(args.processed_csv_dir, "df_ships.csv"))
        df_no_ships = pd.read_csv(os.path.join(args.processed_csv_dir, "df_no_ships.csv"))

        df = pd.concat([df_ships, df_no_ships], ignore_index=True)
        x_train, x_val, y_train, y_val = train_test_split(df['ImageId'], df['HasShip'], test_size=0.2, stratify=df['HasShip'], random_state=42)

        train_loader = classification_data_loader(x_train, y_train, args.dataset_path, batch_size=32)
        validation_loader = classification_validation_data_loader(x_val, y_val, args.dataset_path, batch_size=32)
        # images, labels = next(train_loader)
        # print(images.shape, labels.shape)

        total_samples = len(y_train)
        num_zero = (y_train == 0).sum()
        num_one = (y_train == 1).sum()
        weight_zero = (1 / num_zero) * (total_samples / 2.0)
        weight_one = (1 / num_one) * (total_samples / 2.0)

        logger.info(f"Class weights - Zero: {weight_zero:.4f}, One: {weight_one:.4f}")

        model.compile_model(learning_rate=0.001, weight_zero=0.5, weight_one=0.5)
        model.train(train_loader, validation_loader, epochs=2, df_len=len(df), batch_size=128)

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
    main()