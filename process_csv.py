import logging
import pandas as pd

from src.args.process_csv_args import  process_csv_args
from src.utils.data_preprocessor import preprocess_csv, save_df_to_csv
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# Configure logging for all modules
logging.basicConfig(
    level=logging.INFO,
    format='%(name)s - %(levelname)s - %(message)s',
    force=True
)

logger = logging.getLogger(__name__)

# Uncomment the following line if you want to set debug level for this specific logger
# logger.setLevel(logging.DEBUG)



def main():
    args = process_csv_args()
    df = preprocess_csv(args.csv_file_path)
    logger.debug(f"Result of preprocess_csv: {df}")
    df_ship = df[df['ShipCount'] > 0]
    df_no_ship = df[df['ShipCount'] == 0]

    ratio = args.ship_noship_ratio
    ship_ratio = ratio / (ratio + 1)
    no_ship_ratio = 1 / (ratio + 1)

    num_datapoint = args.num_datapoints

    # df_ship_sampled = df_ship.sample(n=int(num_datapoint*ship_ratio), random_state=args.rand_seed)
    df_ship_sampled, _ = train_test_split(df_ship, train_size=int(num_datapoint*ship_ratio), stratify=df_ship['ShipCount'], random_state=args.rand_seed)
    df_no_ship_sampled = df_no_ship.sample(n=int(num_datapoint*no_ship_ratio), random_state=args.rand_seed)

    df_sampled = pd.concat([df_ship_sampled, df_no_ship_sampled])
    df_sampled = shuffle(df_sampled, random_state=args.rand_seed)

    # df_sampled = df_sampled.sample(frac=1, random_state=args.rand_seed).reset_index(drop=True)
    logger.debug(f"Result of pd.concat: {df_sampled}")
    logger.info(f"Number of rows in df_sampled: {len(df_sampled)}")
    logger.info(f"Number of rows with ship: {len(df_sampled[df_sampled['ShipCount'] > 0])}")
    logger.info(f"Number of rows without ship: {len(df_sampled[df_sampled['ShipCount'] == 0])}")

    ship_count_distribution = df_ship_sampled['ShipCount'].value_counts().sort_index()
    logger.info("Distribution of ShipCount:")
    logger.info(f"ShipCount 0: {len(df_sampled[df_sampled['ShipCount'] == 0])}")
    for ship_count, count in ship_count_distribution.items():
        logger.info(f"ShipCount {ship_count}: {count}")
    save_df_to_csv(df_sampled, args.output_path) 
    

if __name__ == "__main__":
    logger.info("Starting process_csv")
    main()
    logger.info("Completed process_csv")
