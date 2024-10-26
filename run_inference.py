import logging	
import tensorflow as tf
from src.args.run_inference_args import process_csv_args

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

    

if __name__ == "__main__":
    logger.info("Starting inference")

    # Set mixed precision policy
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        main()

    logger.info("Completed inference")

