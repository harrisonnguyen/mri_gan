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
from model import cycle_gan
from model_utils import learning_utils as learning
from model_utils import tfrecord_utils as tfrec

import argparse

#3d model
from model3D import cycle_gan3D

def load_data(serialized_example,feature_list, data_list,feature_size,use_3D):
    if use_3D:
        data = tfrec.tfrecord_parser(serialized_example,feature_list,data_list,feature_size**3)
        #image = tf.reshape(data[0], [feature_size,feature_size,feature_size])

        # get a random slice of the 3d image
        #index = tf.random_uniform((),minval=0,maxval=data[5],dtype=tf.int64)
        #slice = image[index,:,:]
        return tf.reshape(data[0],[feature_size,feature_size,feature_size,1])
    else:
        data = tfrec.tfrecord_parser(serialized_example,feature_list,data_list,feature_size**2)
        #image = tf.reshape(data[0], [feature_size,feature_size,feature_size])

        # get a random slice of the 3d image
        #index = tf.random_uniform((),minval=0,maxval=data[5],dtype=tf.int64)
        #slice = image[index,:,:]
        return tf.reshape(data[0],[feature_size,feature_size,1])

def main(parser):

    # model paramters
    N_BASE_FILTER = parser.n_filter
    N_RESIDUAL_BLOCKS = parser.n_residual_blocks
    PATCH_SIZE = 128
    N_MODALITY = 1
    BATCH_SIZE = parser.batch_size
    N_EPOCHS = parser.n_epoch
    CHECKPOINT_NAME = parser.checkpoint_name
    DATA_DIR = parser.data_dir
    USE_3D = parser.use_3D

    training_ph = tf.placeholder_with_default(False,())
    learning_rate_ph = tf.placeholder_with_default(2e-4,())
    decay_step_ph = tf.placeholder(tf.int32,())
    if USE_3D:
        (x_phA, x_phB, predicted_A,predicted_B,
        real_A,real_B,fake_A,fake_B,reconstructA,reconstructB) = cycle_gan3D(PATCH_SIZE,N_MODALITY,N_BASE_FILTER,N_RESIDUAL_BLOCKS,training_ph)
    else:
        (x_phA, x_phB, predicted_A,predicted_B,
        real_A,real_B,fake_A,fake_B,reconstructA,reconstructB) = cycle_gan(PATCH_SIZE,N_MODALITY,N_BASE_FILTER,N_RESIDUAL_BLOCKS,training_ph)
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
    if USE_3D:
        variables = [genA_loss,discrimB_loss,genB_loss,discrimA_loss,cycle_loss,
            x_phA[:,40,:],x_phB[:,40,:],predicted_B[:,40,:],predicted_A[:,40,:],reconstructA[:,40,:],
             reconstructB[:,40,:]]
    else:
        variables = [genA_loss,discrimB_loss,genB_loss,discrimA_loss,cycle_loss,
            x_phA,x_phB,predicted_B,predicted_A,reconstructA,reconstructB]

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


    """Training data"""
    ## get the control of WMH
    wmh_control = tfrec.get_files_in_dir([os.path.join(DATA_DIR,"WMH/CON/")],".tfrecords")
    bmc_control = tfrec.get_files_in_dir([os.path.join(DATA_DIR,"BMC/CON/")],".tfrecords")
    print(wmh_control)
    #feature_list =  ['data','scanner','gender','class','age','x_shape','y_shape','z_shape']
    #data_list = ['float','int','int','int','int','int','int','int']
    feature_list =  ['data','x_shape','y_shape','z_shape']
    data_list = ['float','int','int','int']
    feature_size = PATCH_SIZE

    ## create the tensorflow queue
    parse_fn = lambda x: load_data(x,feature_list,data_list,feature_size,USE_3D)
    (A_iterator,A_interator_next,filename_ph,batch_ph) = tfrec.create_tfrecord_queue(parse_fn,n_epochs=N_EPOCHS)

    (B_iterator,B_interator_next,filename_ph,batch_ph) = tfrec.create_tfrecord_queue(parse_fn,None,
                                        filename_ph,batch_ph)

    # begin session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    # ability to save/load models
    saver = tf.train.Saver()
    base_dir = os.path.join(home,'tensorflow_checkpoints/cycle_mri/'+CHECKPOINT_NAME+'_filter'
                            +str(N_BASE_FILTER)+'_residual_block'+str(N_RESIDUAL_BLOCKS))


    checkpoint_dir = os.path.join(base_dir ,'train')
    train_writer = tf.summary.FileWriter(checkpoint_dir, sess.graph)

    checkpoint_dir = os.path.join(base_dir,'validation')
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
    sess.run(A_iterator.initializer,feed_dict={filename_ph:wmh_control,
                                        batch_ph:BATCH_SIZE})
    sess.run(B_iterator.initializer,feed_dict={filename_ph:bmc_control,
                                        batch_ph:BATCH_SIZE})
    print("Begin Training")
    while True:
        try:
            A_image = sess.run(A_interator_next)
            B_image = sess.run(B_interator_next)
            # normal cyclegan but with paired images
            data={x_phA: A_image,
                  x_phB: B_image,
                 training_ph:True,
                  decay_step_ph:1000000}
            summary,_,global_step = sess.run([summary_op,solver,genA_global_step],
                    feed_dict=data)

            if global_step % 100 == 0:
                print("At train step %d" %global_step)
                train_writer.add_summary(summary, global_step)
                """
                ## run the testing results
                data={x_phA: test_batch_images[:,:,:,:3],
                      x_phB: test_batch_images[:,:,:,3:],
                      training_ph:False}
                summary,global_step = sess.run([validation_summary,genA_global_step],
                        feed_dict=data)
                validation_writer.add_summary(summary, global_step)
                """
            if global_step % 1000 == 0:
                checkpoint_name = os.path.join(checkpoint_dir+'/checkpoint/', 'model_step'+str(int(global_step))+'.ckpt')
                save_path = saver.save(sess, checkpoint_name)

        except tf.errors.OutOfRangeError:
            # we've run out of data
            break
    print("Finished training. Computing final erros and saving model.")
    # perform end of epoch calculations
    checkpoint_name = os.path.join(checkpoint_dir+'/checkpoint/', 'model_step'+str(int(global_step))+'.ckpt')
    save_path = saver.save(sess, checkpoint_name)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='cycle gan for mri volumes')
    parser.add_argument("--n_epoch", type=int, default=10, help='int number of epochs to train')
    parser.add_argument("--batch_size", type=int, default=1, help='batch size per training step')
    parser.add_argument("--data_dir", default='data/mri_saggital', help='root directory of data')
    parser.add_argument("--checkpoint_name", help='name of directory of checkpoint')
    parser.add_argument("--n_filter", type=int, default=4, help='no of initial base filters')
    parser.add_argument("--n_residual_blocks", type=int, default=1, help='no of residual blocks in generator')
    parser.add_argument('--use_3D', dest='use_3D', action='store_true')
    parser.set_defaults(use_3D=False)
    args = parser.parse_args()
    main(args)
