from __future__ import print_function
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from os.path import expanduser
home = expanduser("~")

import tensorflow as tf
import numpy as np
import os
import pickle

# custom tensorflow functions
from model import generator, discriminator
from model_utils import learning_utils as learning
from model_utils import tfrecord_utils as tfrec

# functions to read images
from matplotlib.pyplot import imread
from skimage.transform import resize

import argparse

def load_train_data(image_path, load_size=140, fine_size=128, is_testing=False):

    # load the image
    img_A = imread(image_path[0])
    img_B = imread(image_path[1])

    if not is_testing:
        img_A = resize(img_A, [load_size, load_size],preserve_range=True)
        img_B = resize(img_B, [load_size, load_size],preserve_range=True)

        h1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
        w1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))

        img_A = img_A[h1:h1+fine_size, w1:w1+fine_size]
        img_B = img_B[h1:h1+fine_size, w1:w1+fine_size]


        if np.random.random() > 0.5:
            img_A = np.fliplr(img_A)
            img_B = np.fliplr(img_B)
    else:
        img_A = resize(img_A, [fine_size, fine_size],preserve_range=True)
        img_B = resize(img_B, [fine_size, fine_size],preserve_range=True)

    # scale between -1 and 1
    img_A = img_A/127.5 - 1.
    img_B = img_B/127.5 - 1.

    img_AB = np.concatenate((img_A, img_B), axis=2)
    # img_AB shape: (fine_size, fine_size, input_c_dim + output_c_dim)
    return img_AB

def main(parser):

    # model paramters
    N_BASE_FILTER = 16
    PATCH_SIZE = 128
    N_MODALITY = 3

    x_phA = tf.placeholder(tf.float32, [None,PATCH_SIZE,PATCH_SIZE,N_MODALITY])
    x_phB = tf.placeholder(tf.float32, [None,PATCH_SIZE,PATCH_SIZE,N_MODALITY])



    training_ph = tf.placeholder_with_default(False,())
    learning_rate_ph = tf.placeholder_with_default(2e-4,())
    decay_step_ph = tf.placeholder(tf.int32,())

    with tf.variable_scope("A_to_B") as scope:
        predicted_B = generator(x_phA,N_BASE_FILTER,training_ph,
                                  output_activation=None)


    with tf.variable_scope("discrimB") as scope:
        real_B = discriminator(x_phB,N_BASE_FILTER,training_ph)
        scope.reuse_variables()
        fake_B = discriminator(predicted_B,N_BASE_FILTER,training_ph)

    with tf.variable_scope("B_to_A") as scope:
        predicted_A = generator(x_phB,N_BASE_FILTER,training_ph,
                                      output_activation=None)

    with tf.variable_scope("discrimA") as scope:
        real_A = discriminator(x_phA,N_BASE_FILTER,training_ph)
        scope.reuse_variables()
        fake_A = discriminator(predicted_A,N_BASE_FILTER,training_ph)

    with tf.variable_scope("B_to_A") as scope:
        scope.reuse_variables()
        reconstructA = generator(predicted_B,N_BASE_FILTER,training_ph,
                                       output_activation=None)

    with tf.variable_scope("A_to_B") as scope:
        scope.reuse_variables()
        reconstructB = generator(predicted_A,N_BASE_FILTER,training_ph,
                                      output_activation=None)

    """
    beginning of loss
    """
    discrimB_loss = (tf.losses.mean_squared_error(predictions=real_B,
                                              labels=tf.ones_like(real_B))
                 + tf.losses.mean_squared_error(predictions=fake_B,
                                                labels=tf.zeros_like(fake_B)))

    discrimA_loss = (tf.losses.mean_squared_error(predictions=real_A,
                                              labels=tf.ones_like(real_A))
                 + tf.losses.mean_squared_error(predictions=fake_A,
                                                labels=tf.zeros_like(fake_A)))
    reconstructA_loss = tf.losses.absolute_difference(x_phA,reconstructA)

    reconstructB_loss = tf.losses.absolute_difference(x_phB,reconstructB)

    cycle_loss = reconstructA_loss + reconstructB_loss

    LAMBDA = 10.0

    genA_loss = (tf.losses.mean_squared_error(predictions=fake_A,
                                                labels=tf.ones_like(fake_A))
                 + LAMBDA*cycle_loss)

    genB_loss = (tf.losses.mean_squared_error(predictions=fake_B,
                                                 labels=tf.ones_like(fake_B))
                  + LAMBDA*cycle_loss)

    reconstruction_loss = tf.losses.mean_squared_error(predictions=predicted_B,
                                                       labels=x_phB)


    """
    Solvers
    """
    genA_solver, genA_global_step = learning.create_solver(genA_loss,
                                                     learning_rate_ph,
                                                    decay_step_ph,
                                                    learning.get_params(['B_to_A']))
    genB_solver, genB_global_step = learning.create_solver(genB_loss,
                                                         learning_rate_ph,
                                                        decay_step_ph,
                                                        learning.get_params(['A_to_B']))
    discrimA_solver, discrimA_global_step = learning.create_solver(discrimA_loss,
                                                         learning_rate_ph,
                                                        decay_step_ph,
                                                        learning.get_params(['discrimA']))
    discrimB_solver, discrimB_global_step = learning.create_solver(discrimB_loss,
                                                         learning_rate_ph,
                                                        decay_step_ph,
                                                        learning.get_params(['discrimB']))
    with tf.control_dependencies(
        [discrimA_solver,discrimB_solver,
        genA_solver,genB_solver]):
        solver = tf.no_op(name='optimisers')

    # tensorboard summary
    variables = [genA_loss,discrimB_loss,genB_loss,discrimA_loss,cycle_loss,
            x_phA,x_phB,predicted_B,predicted_A,reconstructA,
             reconstructB]
    types = ['scalar']*5+['image']*6
    names = ['loss/genA_loss','loss/discrimB_loss','loss/genB_loss','loss/discrimA_loss',
            'loss/cycle_loss','image/x_phA','image/x_phB','image/fake_B','image/fake_A',
            'image/reconstructA','image/reconstructB']

    summary_op,weights_op = learning.create_summary(variables,types,names)

    tf.summary.scalar('validation/accuracy', reconstruction_loss,
                                       collections=['validation'])
    tf.summary.image('validation/real_B', x_phB,
                                       collections=['validation'])
    tf.summary.image('validation/fake_B', predicted_B,
                                       collections=['validation'])
    validation_summary = tf.summary.merge_all(key='validation')

    ## sorting out data
    N_EPOCHS = parser.n_epoch
    BATCH_SIZE = parser.batch_size

    """Training data"""



    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    base_dir = os.path.join(home,'tensorflow_checkpoints/cycle_mri/'+'_filter'
                            +str(N_BASE_FILTER))


    checkpoint_dir = base_dir + 'train'
    train_writer = tf.summary.FileWriter(checkpoint_dir, sess.graph)

    checkpoint_dir = base_dir + 'validation'
    validation_writer = tf.summary.FileWriter(checkpoint_dir, sess.graph)
    try:
        saver.restore(sess,tf.train.latest_checkpoint(checkpoint_dir+'/checkpoint'))
        global_step = sess.run(genA_global_step)
        print("Loaded model at step %d" %global_step)
    except ValueError:
        print("Creating new model")
        global_step = 0

    """
    Training begins
    """

    ## split the training data to match/unmatched data
    training_files_A = np.sort(training_files_A)
    training_files_B = np.sort(training_files_B)

    from sklearn.model_selection import train_test_split
    A_unpair,A_pair,B_unpair,B_pair = train_test_split(training_files_A,training_files_B,
                                                        random_state=42,test_size=PER_SPLIT,shuffle=True)
    print(A_pair[:10])
    print(B_pair[:10])
    # indexs for paired images

    batch_idxs = min(len(A_unpair), len(B_unpair)) // BATCH_SIZE

    # organise the test dataset



    for epoch in range(N_EPOCHS):
        print("At epoch %d" %epoch)
        np.random.shuffle(A_unpair)
        np.random.shuffle(B_unpair)
        #np.random.shuffle(s)
        for idx in range(0, batch_idxs):
            batch_files = list(zip(A_unpair[idx * BATCH_SIZE:(idx + 1) * BATCH_SIZE],
                                   B_unpair[idx * BATCH_SIZE:(idx + 1) * BATCH_SIZE]))
            unpair_batch_images = [load_train_data(batch_file) for batch_file in batch_files]
            unpair_batch_images = np.array(unpair_batch_images).astype(np.float32)

            index = np.random.randint(0,len(A_pair),BATCH_SIZE)
            batch_files = list(zip(np.array(A_pair)[index],np.array(B_pair)[index]))
            pair_batch_images = [load_train_data(batch_file) for batch_file in batch_files]
            pair_batch_images = np.array(pair_batch_images).astype(np.float32)
            if USE_SEMI:

                data={x_phA: unpair_batch_images[:,:,:,:3],
                      x_phB: unpair_batch_images[:,:,:,3:],
                      x_phA_paired: pair_batch_images[:,:,:,:3],
                      x_phB_paired: pair_batch_images[:,:,:,3:],
                     training_ph:True,
                      decay_step_ph:1000000}
                summary,_,_,global_step = sess.run([summary_op,solver,discrim_pair_solver,genA_global_step],
                        feed_dict=data)
            else:
                # normal cyclegan but with paired images
                data={x_phA: unpair_batch_images[:,:,:,:3],
                      x_phB: unpair_batch_images[:,:,:,3:],
                     training_ph:True,
                      decay_step_ph:1000000}
                summary,_,global_step = sess.run([summary_op,solver,genA_global_step],
                        feed_dict=data)

                data={x_phA: pair_batch_images[:,:,:,:3],
                      x_phB: pair_batch_images[:,:,:,3:],
                     training_ph:True,
                      decay_step_ph:1000000}
                summary,_,global_step = sess.run([summary_op,solver,genA_global_step],
                        feed_dict=data)

            if global_step % 100 == 0:
                train_writer.add_summary(summary, global_step)

                # load the test data
                index = np.random.randint(0,len(test_files_A),50)
                batch_files = list(zip(np.array(test_files_A)[index],np.array(test_files_B)[index]))
                test_batch_images = [load_train_data(batch_file) for batch_file in batch_files]
                test_batch_images = np.array(pair_batch_images).astype(np.float32)

                ## run the testing results
                data={x_phA: test_batch_images[:,:,:,:3],
                      x_phB: test_batch_images[:,:,:,3:],
                      training_ph:False}
                summary,global_step = sess.run([validation_summary,genA_global_step],
                        feed_dict=data)
                validation_writer.add_summary(summary, global_step)
            checkpoint_name = os.path.join(checkpoint_dir+'/checkpoint/', 'model_step'+str(int(global_step))+'.ckpt')
            save_path = saver.save(sess, checkpoint_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='cycle gan for mri volumes')
    parser.add_argument("--n_epoch", type=int, default=10, help='int number of epochs to train')
    parser.add_argument("--batch_size", type=int, default=1, help='int number of epochs to train')
    args = parser.parse_args()
    main(args)
