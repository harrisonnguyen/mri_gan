import tensorflow as tf
import pandas as pd
from model.cyclegan import CycleGAN3D
from load_mri_data import load_data


def main():

    df = pd.read_csv("data/demographics.csv")
    df["Filename"] = df["Filename"].str.replace('.nii',".tfrecords")
    df["Filename"] = "data/mri_full/" + df["Filename"].astype(str)
    image_size = [1,121,145,121,1]
    ksizes = [1,64,64,64,1]
    strides = [1,32,32,32,1]
    gan = CycleGAN3D(ksizes[1:],depth=2)
    dataset = load_data(df["Filename"].values[:5],
                        image_size=image_size,
                        batch_size=1,
                        ksizes=ksizes,
                        strides=strides)
    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
    next_image = tf.squeeze(iterator.get_next(),axis=0)
    sess = tf.keras.backend.get_session()
    img_A = sess.run(next_image)
    img_B = sess.run(next_image)
    gan.train_step(img_A,img_B)

if __name__ == "__main__":
    main()
