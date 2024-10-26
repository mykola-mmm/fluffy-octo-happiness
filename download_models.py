import os
import logging
import gdown
from src.args.download_models_args import process_download_args

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_model(url, output_path):
    """
    Download a model from Google Drive
    
    Args:
        file_id (str): Google Drive file ID
        output_path (str): Path where the model should be saved
    """
    
    try:
        gdown.download(url, output_path, quiet=False)
        logger.info(f"Successfully downloaded model to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error downloading model: {str(e)}")
        return False

def main():
    # Parse arguments
    args = process_download_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define output paths
    classification_model_path = os.path.join(args.output_dir, "classification_model.keras")
    segmentation_model_path = os.path.join(args.output_dir, "segmentation_model.keras")
    
    # Download classification model
    logger.info("Downloading classification model...")
    if download_model(args.classification_model_url, classification_model_path):
        logger.info("Classification model downloaded successfully")
    else:
        logger.error("Failed to download classification model")
        return
    
    # Download segmentation model
    logger.info("Downloading segmentation model...")
    if download_model(args.segmentation_model_url, segmentation_model_path):
        logger.info("Segmentation model downloaded successfully")
    else:
        logger.error("Failed to download segmentation model")
        return
    
    logger.info("All models downloaded successfully!")

if __name__ == "__main__":
    main()

