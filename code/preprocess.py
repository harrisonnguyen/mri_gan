import tensorflow as tf
import nibabel as nib
import pandas as pd
import click
import os
import numpy as np

def write_example(image,directory,file_name):
    writer = tf.python_io.TFRecordWriter(os.path.join(directory,file_name +".tfrecords"))
    temp_dict = {}
    temp_dict['img'] = tf.train.Feature(float_list=tf.train.FloatList(value = image))

    #construct the example proto object
    example = tf.train.Example(
                features = tf.train.Features(
                                feature = temp_dict))
    # use the proto object to serialize the example to a string
    serialized = example.SerializeToString()
    writer.write(serialized)
    writer.close()
    return serialized

@click.command()
@click.option('--data-dir',
            default="/media/harrison/ShortTerm/Users/HarrisonG/research/rich_data/data",
            type=click.Path(
                    file_okay=False,
                    dir_okay=True,
                    writable=False),
             help="directory to save results",
             show_default=True)


def main(data_dir):
    df = pd.read_csv("data/demographics.csv")
    for ele in df["Filename"].values[:5]:
        epi_img = nib.load(os.path.join(data_dir,"mwc1"+ele))
        img_data = epi_img.get_fdata()
        print(img_data.shape)
        epi_img.uncache()
        write_example(np.reshape(np.array(img_data),(-1)),'data/mri_full/',ele.split(".")[0])

if __name__ == "__main__":
    exit(main())
