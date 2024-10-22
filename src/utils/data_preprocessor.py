import pandas as pd
import logging
import os

logger = logging.getLogger(__name__)

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
