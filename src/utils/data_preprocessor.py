import pandas as pd
import logging
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def preprocess_csv(csv_file_path):
    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(csv_file_path)

        logger.info(f"Successfully read CSV file: {csv_file_path}")

        # Create HasShip column based on EncodedPixels
        df['HasShip'] = df['EncodedPixels'].notna().astype(int)
        # Create ShipCount column
        df['ShipCount'] = df.groupby('ImageId')['HasShip'].transform('sum')
        # Create AllEncodedPixels column
        df['AllEncodedPixels'] = df.groupby('ImageId')['EncodedPixels'].transform(lambda x: ' '.join(x.dropna()))
        # Drop HasShip and EncodedPixels columns
        df = df.drop(columns=['EncodedPixels'])
        # Drop duplicates based on ImageId and keep the first unique value
        df = df.drop_duplicates(subset=['ImageId'], keep='first')

        return df
    except FileNotFoundError:
        logger.error(f"CSV file not found: {csv_file_path}")
        return None
    except pd.errors.EmptyDataError:
        logger.error(f"CSV file is empty: {csv_file_path}")
        return None
    except Exception as e:
        logger.error(f"Error reading CSV file {csv_file_path}: {str(e)}")
        return None

def save_df_to_csv(df, output_path):
    # Create the directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created directory: {output_dir}")
    
    df.to_csv(output_path, index=False)
    logger.info(f"Processed CSV saved to: {output_path}")

def balanced_sample(df, num_datapoints, ship_count_col='ShipCount', random_state=None):
    unique_ship_counts = sorted(df[ship_count_col].unique(), reverse=True)
    num_classes = len(unique_ship_counts)
    samples_per_class = num_datapoints // num_classes

    logger.debug(f"Number of classes: {unique_ship_counts}")

    sampled_dfs = []
    for i, ship_count in enumerate(unique_ship_counts):
        total_sampled_dfs = 0
        for y in sampled_dfs:
            total_sampled_dfs += len(y)
        samples_per_class = (num_datapoints - total_sampled_dfs) // (num_classes - i)
        class_df = df[df[ship_count_col] == ship_count]
        if len(class_df) < samples_per_class:
            sampled_dfs.append(class_df)
        else:
            sampled_dfs.append(class_df.sample(n=samples_per_class, random_state=random_state))
        logger.debug(f"i - {i} ## ship_count - {ship_count} ## samples_per_class - {samples_per_class} ## len(sampled_dfs) - {num_datapoints - total_sampled_dfs}")
    return pd.concat(sampled_dfs).sample(frac=1, random_state=random_state).reset_index(drop=True)
