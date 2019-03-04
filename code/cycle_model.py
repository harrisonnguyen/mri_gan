from __future__ import print_function
import os
import tensorflow as tf
import numpy as np

from model import generator, discriminator
from model_utils import learning_utils as learning


class CycleGan(object):
    def __init__(self,
                n_filters,
                n_residual_blocks,
                patch_size,
                n_modality,
                base_dir,
                cycle_loss_weight=10.0):
        self.n_filters = n_filters
        self.n_residual_blocks = n_residual_blocks
        self.patch_size = patch_size
        self.n_modality = n_modality
        self._LAMBDA = cycle_loss_weight
        tf.reset_default_graph()
        self._build_graph()
        self._create_loss()
        self._create_summary()
        self._create_optimiser()
        self._graph = tf.get_default_graph()


        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        self.sess = tf.Session(config=config,graph=self._graph)
        self.sess.run(tf.global_variables_initializer())

        self._saver = tf.train.Saver()
        checkpoint_dir = os.path.join(base_dir,'train')
        self._train_writer = tf.summary.FileWriter(checkpoint_dir, self.sess.graph)

        checkpoint_dir = os.path.join(base_dir,'validation')
        self._validation_writer = tf.summary.FileWriter(checkpoint_dir, self.sess.graph)

        self.checkpoint_dir = os.path.join(base_dir,'checkpoint')

    def _build_graph(self):
        self._xphA = tf.placeholder(tf.float32, [None,self.patch_size,self.patch_size,self.n_modality])
        self._xphB = tf.placeholder(tf.float32, [None,self.patch_size,self.patch_size,self.n_modality])

        self._training_ph = tf.placeholder_with_default(False,())

        #begin_learning_rate_decay_ph = tf.placeholder_with_default(False,())

        self._model_global_step = tf.Variable(0,trainable=False,dtype=tf.int32)
        self._model_global_step_inc = tf.assign_add(self._model_global_step,1)
        self._epoch = tf.Variable(0,trainable=False,dtype=tf.int32)
        self._epoch_inc = tf.assign_add(self._epoch,1)

        with tf.variable_scope("genA") as scope:
            self._predictedA = generator(self._xphB,
                                self.n_filters,
                                self._training_ph,
                                tf.nn.tanh,
                                self.n_modality)

        with tf.variable_scope("genB") as scope:
            self._predictedB = generator(self._xphA,
                                self.n_filters,
                                self._training_ph,
                                tf.nn.tanh,
                                self.n_modality)

            scope.reuse_variables()
            self._reconstructB = generator(self._predictedA,
                                            self.n_filters,
                                            self._training_ph,
                                            tf.nn.tanh,
                                            self.n_modality)
        with tf.variable_scope("genA") as scope:
            scope.reuse_variables()
            ## the cycle for gen A
            self._reconstructA = generator(self._predictedB,
                                    self.n_filters,
                                    self._training_ph,
                                    tf.nn.tanh,
                                    self.n_modality)

        with tf.variable_scope("discriminatorA") as scope:
            self._fakeA,_ = discriminator(self._predictedA,
                                    self.n_filters,
                                    self._training_ph)
            scope.reuse_variables()
            self._realA,_ = discriminator(self._xphA,
                                    self.n_filters,
                                    self._training_ph)

        with tf.variable_scope("discriminatorB") as scope:
            self._fakeB,_ = discriminator(self._predictedB,
                                    self.n_filters,
                                    self._training_ph)
            scope.reuse_variables()
            self._realB,_ = discriminator(self._xphB,
                                    self.n_filters,
                                    self._training_ph)
    def _create_loss(self):
        self._discrimB_loss = (tf.losses.mean_squared_error(predictions=self._realB,
                                                  labels=tf.ones_like(self._realB))
                     + tf.losses.mean_squared_error(predictions=self._fakeB,
                                                    labels=tf.zeros_like(self._fakeB)))

        self._discrimA_loss = (tf.losses.mean_squared_error(predictions=self._realA,
                                                  labels=tf.ones_like(self._realA))
                     + tf.losses.mean_squared_error(predictions=self._fakeA,
                                                    labels=tf.zeros_like(self._fakeA)))
        self._reconstructA_loss = tf.losses.absolute_difference(self._xphA,self._reconstructA)

        self._reconstructB_loss = tf.losses.absolute_difference(self._xphB,self._reconstructB)

        self._cycle_loss = self._reconstructA_loss + self._reconstructB_loss



        self._genA_loss = (tf.losses.mean_squared_error(predictions=self._fakeA,
                                    labels=tf.ones_like(self._fakeA))
                     + self._LAMBDA*self._cycle_loss)

        self._genB_loss = (tf.losses.mean_squared_error(predictions=self._fakeB,
                                                     labels=tf.ones_like(self._fakeB))
                      + self._LAMBDA*self._cycle_loss)

        self._reconstruction_loss = tf.losses.mean_squared_error(predictions=self._predictedB,
                                                labels=self._xphB)

    def _create_optimiser(self):
        self._learning_rate_ph = tf.placeholder_with_default(2e-4,())
        self._decay_step_ph = tf.placeholder_with_default(100,())
        self._training_decay_step = tf.Variable(0,trainable=False,dtype=tf.int32)

        genA_solver = learning.create_solver(self._genA_loss,
                                            self._training_decay_step,
                                             self._learning_rate_ph,
                                            self._decay_step_ph,
                                            learning.get_params(['genA']),
                                            increment_global_step=False,
                                            end_learning_rate=0.0)
        genB_solver= learning.create_solver(self._genB_loss,
                                            self._training_decay_step,
                                             self._learning_rate_ph,
                                            self._decay_step_ph,
                                            learning.get_params(['genB']),
                                            increment_global_step=False,
                                            end_learning_rate=0.0)
        discrimA_solver= learning.create_solver(self._discrimA_loss,
                                                self._training_decay_step,
                                                 self._learning_rate_ph/2.0,
                                                self._decay_step_ph,
                                                learning.get_params(['discriminatorA']),
                                                increment_global_step=False,
                                                end_learning_rate=0.0)
        discrimB_solver = learning.create_solver(self._discrimB_loss,
                                                self._training_decay_step,
                                                 self._learning_rate_ph/2.0,
                                                self._decay_step_ph,
                                                learning.get_params(['discriminatorB']),
                                                increment_global_step=False,
                                                end_learning_rate=0.0)
        with tf.control_dependencies(
            [discrimA_solver,discrimB_solver,
            genA_solver,genB_solver]):
            self._solver = tf.no_op(name='optimisers')

    def _create_summary(self):
        variables = [self._genA_loss,self._discrimB_loss,self._genB_loss,self._discrimA_loss,self._cycle_loss,
                self._xphA,self._xphB,self._predictedB,self._predictedA,self._reconstructA,
                 self._reconstructB]
        types = ['scalar']*5+['image']*6
        names = ['loss/genA_loss','loss/discrimB_loss','loss/genB_loss','loss/discrimA_loss',
                'loss/cycle_loss','image/x_phA','image/x_phB','image/fake_B','image/fake_A',
                'image/reconstructA','image/reconstructB']
        self._summary_op,self._weights_op = learning.create_summary(variables,types,names)

        tf.summary.scalar('validation/accuracy', self._reconstruction_loss,
                                           collections=['validation'])
        tf.summary.image('validation/real_B', self._xphB,
                                           collections=['validation'])
        tf.summary.image('validation/fake_B', self._predictedB,
                                           collections=['validation'])
        self._validation_summary = tf.summary.merge_all(key='validation')

    def train_step(self,A,B,
                        learning_rate=2e-4,
                        write_summary=False):
        data={self._xphA: A,
              self._xphB: B,
              self._training_ph:True,
              self._learning_rate_ph:learning_rate}
        summary,global_step,_,_ =self.sess.run([self._summary_op,
                                            self._model_global_step,
                                            self._solver,
                                            self._model_global_step_inc],
                                            feed_dict=data)
        if write_summary:
            self._train_writer.add_summary(summary, global_step)
        return global_step

    def validate(self,A,B):
        data={self._xphA: A,
              self._xphB: B,
              self._training_ph:False}

        summary,global_step = self.sess.run([self._validation_summary,
                                        self._model_global_step],
                                            feed_dict=data)
        self._validation_writer.add_summary(summary, global_step)

        return global_step

    def save_checkpoint(self):
        epoch = self.get_epoch()
        self._saver.save(self.sess, os.path.join(self.checkpoint_dir,"epoch"+str(epoch)+".ckpt"))
        return epoch

    def increment_epoch(self):
        epoch = self.sess.run(self._epoch_inc)
        return epoch

    def get_epoch(self):
        return self.sess.run(self._epoch)

    def restore_latest_checkpoint(self):
        self._saver.restore(self.sess,tf.train.latest_checkpoint(self.checkpoint_dir))
        return self.sess.run(self._model_global_step)

    def score(self,A,B):
        """
        The score will be determined how well A is transformed to B
        """
        data={self._xphA: A,
              self._xphB: B,
              self._training_ph:False}
        return self.sess.run(self._reconstruction_loss,
                        feed_dict=data)
    def transform_to_A(self,B):
        data={self._xphB: B,
              self._training_ph:False}
        return self.sess.run(self._predictedA,
                        feed_dict=data)
    def transform_to_B(self,A):
        data={self._xphA: A,
              self._training_ph:False}
        return self.sess.run(self._predictedB,
                        feed_dict=data)
