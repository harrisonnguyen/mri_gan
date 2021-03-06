import tensorflow as tf
import pandas as pd
from model.cyclegan import CycleGAN
from load_mri_data import load_data
import click
import numpy as np
import csv
import os
def index_gen(n_slices,batch_size):
    index = np.array(range(n_slices))
    np.random.shuffle(index)
    for i in range(int(np.ceil(n_slices/batch_size))):
        items = index[i*batch_size:(i+1)*batch_size]
        yield items
def write_params(params):
    w = csv.writer(open(
                    os.path.join(params["checkpoint_dir"],
                                "config.csv"),
                    "w+"))
    for key, val in params.items():
        w.writerow([key, val])
    return
@click.command()
@click.option('--checkpoint-dir',
            type=click.Path(
                    file_okay=False,
                    dir_okay=True,
                    writable=True),
             help="directory to save results",
             show_default=True)
@click.option('--data-dir',
            default="data/mri_full/",
            type=click.Path(
                    file_okay=False,
                    dir_okay=True,
                    writable=False),
             help="directory of data",
             show_default=True)
@click.option('--gf',
            default=32,
            type=click.INT,
             help="Number of initial filters in generator",
             show_default=True)
@click.option('--df',
            default=64,
            type=click.INT,
             help="Number of initial filters in discriminator",
             show_default=True)
@click.option('--depth',
            default=3,
            type=click.INT,
             help="Number of convolution blocks for discirm and gen",
             show_default=True)
@click.option('--patch-size',
            default=128,
            type=click.INT,
             help="Size of image",
             show_default=True)
@click.option('--n-channels',
            default=1,
            type=click.INT,
             help="Number of channels",
             show_default=True)
@click.option('--cycle-loss-weight',
            default=10.0,
            type=click.FLOAT,
             help="Relative loss of cycle",
             show_default=True)
@click.option('--learning-rate',
            default=2e-4,
            type=click.FLOAT,
             help="Initial learning rate",
             show_default=True)
@click.option('--batch-size',
            default=16,
            type=click.INT,
             help="batch size for training",
             show_default=True)
@click.option('--n-epochs',
            default=20,
            type=click.INT,
             help="enumber of epochs for training",
             show_default=True)
@click.option('--summary-freq',
            default=500,
            type=click.INT,
             help="enumber of epochs for training",
             show_default=True)
@click.option('--end-learning-rate',
            default=2e-6,
            type=click.FLOAT,
             help="ending learning rate to decay to",
             show_default=True)
@click.option('--begin-decay',
            default=100,
            type=click.INT,
             help="epochs to begin decay",
             show_default=True)
@click.option('--decay-steps',
            default=100,
            type=click.INT,
             help="number of epochs to decay learning rate",
             show_default=True)
def main(checkpoint_dir,
        data_dir,
        gf,df,depth,patch_size,n_channels,
        cycle_loss_weight,learning_rate,batch_size,n_epochs,summary_freq,end_learning_rate,begin_decay,decay_steps):

    data = pd.read_csv("data/demographics.csv")
    data["Filename"] = data["Filename"].str.replace('.nii',".tfrecords")
    data["Filename"] = data_dir + data["Filename"].astype(str)
    image_size = [121,145,121,1]
    gan = CycleGAN(checkpoint_dir,
                    gf=gf,
                    df=df,
                    depth=depth,
                    patch_size=patch_size,
                    n_modality=n_channels,
                    cycle_loss_weight=cycle_loss_weight,
                    initial_learning_rate=learning_rate,
                    begin_decay=begin_decay,
                    end_learning_rate=end_learning_rate,
                    decay_steps=decay_steps)
    set_A = data[(data["Scanner"]=="WMH") & (data["Class"]=="CON")]
    set_B = data[(data["Scanner"]=="NRA") & (data["Class"]=="CON")]
    params = click.get_current_context().params
    write_params(params)

    dataset_A = load_data(set_A["Filename"].values,
                        image_size=image_size,
                        is_3D=False,
                        buffer_size=batch_size*3)
    dataset_B = load_data(set_B["Filename"].values,
                        image_size=image_size,
                        is_3D=False,
                        buffer_size=batch_size*3)

    sess = gan.sess
    iterator_A = tf.compat.v1.data.make_initializable_iterator(dataset_A)
    iterator_B = tf.compat.v1.data.make_initializable_iterator(dataset_B)
    next_A = iterator_A.get_next()
    next_B = iterator_B.get_next()
    try:
        i = gan.restore_latest_checkpoint()
        print("Restoring at step {}".format(i))
    except:
        i = 0
        print("Creating new model")
    for k in range(n_epochs):
        sess.run(iterator_A.initializer)
        sess.run(iterator_B.initializer)
        while True:
            try:
                img_A = sess.run(next_A)
                img_B = sess.run(next_B)
                gen = index_gen(min(img_A.shape[0],img_B.shape[0]),batch_size)
                for ele in gen:
                    write_summary = i%summary_freq == 0
                    epoch = gan.train_step(img_A[ele],
                                            img_B[ele],
                                            write_summary=write_summary)
                    i+=1

            except tf.errors.OutOfRangeError:
                current_epoch = gan.increment_epoch()
                gan.save_checkpoint()
                print("finished epoch %d. Saving checkpoint" %current_epoch)
                break


if __name__ == "__main__":
    main()
