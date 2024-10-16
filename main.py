import sys
import os

# from src.data_loader import load_data
# from src.model.segmentation import UNet
# from src.model.classification import ClassifierModel
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
    if args.task == "train":
        logger.info("Starting training task...")
        model = UNet()
        model.summary()

        
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