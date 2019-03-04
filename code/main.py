from __future__ import print_function
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from os.path import expanduser

import tensorflow as tf
import load_data
import argparse
import os
import numpy as np
from model_utils import tfrecord_utils as tfrecord

from cycle_model import CycleGan

class CycleGanExperiment(object):
    def __init__(self,
                n_filters,
                n_residual_blocks,
                patch_size,
                initial_learning_rate,
                data_dir,
                n_epochs,
                batch_size,
                home_dir,
                is_semi_supervised,
                setA,
                setB):
        self.n_modality = 1
        self.n_filters = n_filters
        self.n_residual_blocks = n_residual_blocks
        self.patch_size = patch_size
        self.initial_learning_rate = initial_learning_rate
        self.setA = setA
        self.setB = setB

        self.data_dir = data_dir


        ## training parameters
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        if home_dir == "~":
            self.home_dir = expanduser(home_dir)
        else:
            self.home_dir = home_dir

        self.base_dir = os.path.join(self.home_dir,"tensorflow_checkpoints/mri_gan/"+setA.replace("/","")+"_" + setB.replace("/",""))

        self.model = self._build_model()

        try:
            self.global_step = self.model.restore_latest_checkpoint()
            print("Loaded model at step %d" %self.global_step)
        except ValueError:
            print("Creating new model")
            self.global_step = 0

        self._load_data(is_semi_supervised)

    def _build_model(self):
        self.base_dir = os.path.join(self.base_dir,'cycle_gan')
        return CycleGan(self.n_filters,
                        self.n_residual_blocks,
                        self.patch_size,
                        self.n_modality,
                        self.base_dir)

    def _load_data(self,is_semi_supervised):
        data_dir = self.data_dir
        self.data_dirA = os.path.join(data_dir,self.setA)
        self.data_dirB = os.path.join(data_dir,self.setB)
        self.data_queue = load_data.load_data(
                            self.data_dirA,
                            self.data_dirB,
                            self.data_dirA,
                            self.data_dirB,
                            is_semi_supervised=is_semi_supervised,
                            paired_train_dir=None
                            )
        self.A_test = np.sort(tfrecord.get_files_in_dir([self.data_dirA]))
        self.B_test = np.sort(tfrecord.get_files_in_dir([self.data_dirB]))

    def get_learning_rate(self,
                        current_epoch,
                        begin_decrease,
                        end_decrease,
                        end_learning_rate=2e-8):
        if current_epoch < begin_decrease:
            decayed_learning_rate =  self.initial_learning_rate
        elif current_epoch > end_decrease:
            decayed_learning_rate = end_learning_rate
        else:
            # calculate a linear learning rate decay
            gradient = (self.initial_learning_rate - end_learning_rate)/(begin_decrease-end_decrease)
            intercept = self.initial_learning_rate - gradient*begin_decrease
            decayed_learning_rate = gradient*current_epoch + intercept
        return decayed_learning_rate


    def run(self,epoch_decrease=15,epoch_end=40,
                tensorboard_update=500):
        self.data_queue['validation'].initialise_queue(self.batch_size*50,
                                                    self.model.sess)
        if 'paired' in self.data_queue.keys():
            self.data_queue['paired'].initialise_queue(self.batch_size,
                                                        self.model.sess)

        self.current_epoch = self.model.get_epoch()
        print("At epoch {0}".format(self.current_epoch))
        while self.current_epoch < self.n_epochs:
            learning_rate = self.get_learning_rate(
                                self.current_epoch,epoch_decrease,epoch_end)
            print(learning_rate)
            self.data_queue['unpaired'].initialise_queue(self.batch_size,self.model.sess)
            while True:
                try:
                    if self.global_step % tensorboard_update == 0:
                        A,B =  self.data_queue['validation'].get_next()
                        self.model.validate(A,B)
                        write_summary = True
                    else:
                        write_summary = False
                    self.global_step = self._train_step(learning_rate=learning_rate,
                                    write_summary=write_summary)
                except tf.errors.OutOfRangeError:
                    self.current_epoch = self.model.increment_epoch()
                    self.current_epoch = self.model.save_checkpoint()
                    print("finished epoch %d. Saving checkpoint" %self.current_epoch)
                    break
    def evaluate(self):

        test_queue = load_data.DataQueue(
                    self.A_test,
                    self.B_test,
                    self.dataset,
                    n_epochs=1,
                    is_validation=True,queue_size=5)
        test_queue.initialise_queue(self.batch_size*50,
                                    self.model.sess)
        i = 0
        reconstruction_loss = 0
        while True:
            try:
                A,B =  test_queue.get_next()
                reconstruction_loss += self.model.score(A,B)
                i+=1
            except tf.errors.OutOfRangeError:
                print("finished evaluating model %s with dataset %s with loss %.5f" %(type(self.model),
                            self.dataset,reconstruction_loss/i))
                break
        return reconstruction_loss/i

    def transform_image(self,index,transform_to_B=True):
        """Take an index of the test set and transform it
            Return:
                a n_slice x feature_size x feature_size np.array
                for input and output
        """
        test_queue = load_data.DataQueue(
                    [self.A_test[index]],
                    [self.B_test[index]],
                    self.dataset,
                    n_epochs=1,
                    is_validation=True,queue_size=1)
        test_queue.initialise_queue(self.batch_size,
                                    self.model.sess)
        input_array = []
        output_array = []
        if transform_to_B:
            input_file_name = self.A_test[index]
        else:
            input_file_name = self.B_test[index]
        while True:
            try:
                A,B =  test_queue.get_next()
                if transform_to_B:
                    input_image = A
                    image = self.model.transform_to_B(input_image)
                else:
                    input_image = B
                    image = self.model.transform_to_A(input_image)
                input_array.append(input_image)
                output_array.append(image)
            except tf.errors.OutOfRangeError:
                print("finished transforming %s" %input_file_name)
                break
        return {'input': np.squeeze(np.array(input_array)),
                 'output': np.squeeze(np.array(output_array))}

    def _train_step(self,learning_rate,
                        write_summary):
        A,B =  self.data_queue['unpaired'].get_next()
        global_step = self.model.train_step(A,B,
                    learning_rate=learning_rate,
                    write_summary=write_summary)
        return global_step

def main(parser):
    n_filters = parser.n_filters
    n_residual_blocks = parser.n_residual_blocks
    patch_size = 128
    initial_learning_rate = parser.initial_learning_rate
    data_dir = parser.data_dir
    setA = parser.setA
    setB = parser.setB


    ## training parameters
    n_epochs = parser.n_epoch
    batch_size = parser.batch_size
    if parser.home_dir == "~":
        home_dir = expanduser(parser.home_dir)
    else:
        home_dir = parser.home_dir
    experiment = CycleGanExperiment(
                    n_filters=n_filters,
                    n_residual_blocks=n_residual_blocks,
                    patch_size=patch_size,
                    initial_learning_rate=initial_learning_rate,
                    data_dir=data_dir,
                    n_epochs=n_epochs,
                    batch_size=batch_size,
                    home_dir=home_dir,
                    is_semi_supervised=False,
                    setA=setA,
                    setB=setB)

    experiment.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='unsupervised cyclegan for multi center mri')
    parser.add_argument("--n_epoch", type=int, default=10, help='int number of epochs to train')
    parser.add_argument("--batch_size", type=int, default=1, help='int number of epochs to train')
    parser.add_argument("--n_filters", type=int, default=16, help='no of filters for initial layer of the generator')
    parser.add_argument("--n_residual_blocks", type=int, default=3, help='no of residual layers for generator')
    parser.add_argument("--initial_learning_rate",type=float,default=2e-4, help='initial learning rate')
    parser.add_argument("--home_dir",default="~", help='directory to save tensorflow checkpoints')
    parser.add_argument("--data_dir", default="../data/",help="directory of data")
    parser.add_argument("--setA",help="relative directory (to data_dir) name of modality A")
    parser.add_argument("--setB", help="relative directory (to data_dir) name of modality B")
    args = parser.parse_args()
    main(args)
