import pandas as pd
import tensorflow as tf

def _parse_image_function(example_proto):
    image_feature_description = {
    'img': tf.FixedLenFeature([121,145,121,1], tf.float32),
    }
    # Parse the input tf.Example proto using the dictionary above.
    return tf.parse_single_example(example_proto, image_feature_description)

def augment(image):
    image = image['img']/tf.reduce_max(image['img'])
    paddings = tf.constant([[1,0], [1,0,],[1,0,],[0,0]])
    image = tf.pad(image,paddings,"CONSTANT")
    return image

def load_data(filenames,batch_size):
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(_parse_image_function)
    dataset = dataset.map(augment)
    dataset = dataset.batch(batch_size)
    dataset= dataset.repeat(1)
    return dataset


def main():


    df = pd.read_csv("data/demographics.csv")
    df["Filename"] = df["Filename"].str.replace('.nii',".tfrecords")
    df["Filename"] = "data/mri_full/" + df["Filename"].astype(str)

    dataset = load_data(df["Filename"].values[:5])
    iterator = tf.data.make_one_shot_iterator(dataset)
    sess = tf.Session()
    while True:
        try:
            image = sess.run(iterator.get_next())
        except tf.errors.OutOfRangeError:
            break

if __name__ == "__main__":
    main()
