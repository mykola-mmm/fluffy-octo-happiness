import os
import pandas as pd
import tensorflow as tf


def classification_data_loader(df_x, df_y, dataset_path, batch_size=32):
    df = pd.concat([df_x, df_y], axis=1)
    df = df.sample(frac=1,random_state=42).reset_index(drop=True)
    # print(df)
    while True:
        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i:i+batch_size]
            images = []
            labels = []
            for _, row in batch_df.iterrows():
                image_path = os.path.join(dataset_path, f"{row['ImageId']}")
                image = tf.io.read_file(image_path)
                image = tf.image.decode_jpeg(image, channels=3)
                image = tf.cast(image, tf.float32) / 255.0
                images.append(image)
                labels.append(row['HasShip'])
            yield tf.stack(images), tf.stack(labels)

def classification_validation_data_loader(df_x, df_y, dataset_path, batch_size=32):
    while True:
        x, y = next(classification_data_loader(df_x, df_y, dataset_path, batch_size))
        yield (x, y)