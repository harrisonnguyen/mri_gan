import tensorflow as tf
import pandas as pd
from model.cyclegan import CycleGAN
from load_mri_data import load_data


def main():
    gan = CycleGAN((122,146,122,1),depth=2)
    df = pd.read_csv("data/demographics.csv")
    df["Filename"] = df["Filename"].str.replace('.nii',".tfrecords")
    df["Filename"] = "data/mri_full/" + df["Filename"].astype(str)

    dataset = load_data(df["Filename"].values[:5],batch_size=1)
    iterator = tf.data.make_one_shot_iterator(dataset)
    sess = tf.keras.backend.get_session()
    img_A = sess.run(iterator.get_next())
    img_B = sess.run(iterator.get_next())
    gan.train_step(img_A,img_B)

if __name__ == "__main__":
    main()
