from __future__ import print_function, division
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate,LeakyReLU,UpSampling3D, Conv3D, Conv2D, UpSampling2D,Lambda
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.models import Sequential, Model
from keras import backend as K

from model.unet import unet_model_3d
from utils.instance_norm import InstanceNormalization
from model_utils import learning_utils as learning
import os
class CycleGAN(object):
    def __init__(self,
                base_dir,
                gf=32,
                df=64,
                depth=3,
                patch_size=128,
                n_modality=1,
                cycle_loss_weight=10.0,
                initial_learning_rate=2e-4):
        self.img_shape = [patch_size,patch_size,n_modality]
        self._LAMBDA = cycle_loss_weight
        self.initial_learning_rate = initial_learning_rate
        tf.reset_default_graph()
        self._build_graph(gf,df,depth,patch_size,n_modality)
        self._create_loss()
        self._create_summary()
        self._create_optimiser()


        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        self.sess = tf.keras.backend.get_session()
        self.sess.config = config
        self.sess.run(tf.global_variables_initializer())

        self._saver = tf.train.Saver(save_relative_paths=True)
        checkpoint_dir = os.path.join(base_dir,'train')
        self._train_writer = tf.summary.FileWriter(checkpoint_dir, self.sess.graph)

        checkpoint_dir = os.path.join(base_dir,'validation')
        self._validation_writer = tf.summary.FileWriter(checkpoint_dir, self.sess.graph)

        self.checkpoint_dir = os.path.join(base_dir,'checkpoint')

    def _build_graph(self,gf,df,depth,patch_size,n_modality):
        self._xphA = tf.placeholder(tf.float32,
                                    [None,patch_size,patch_size,n_modality])
        self._xphB = tf.placeholder(tf.float32,
                                    [None,patch_size,patch_size,n_modality])


        self._batch_step = tf.Variable(0,trainable=False,dtype=tf.int32)
        self._batch_step_inc = tf.assign_add(self._batch_step,1)
        self._epoch = tf.Variable(0,trainable=False,dtype=tf.int32)
        self._epoch_inc = tf.assign_add(self._epoch,1)

        self.g_AB = self.build_generator(gf,depth)
        self.g_BA = self.build_generator(gf,depth)

        # translate images to new domain
        self._predictedB = self.g_AB(self._xphA)
        self._predictedA = self.g_BA(self._xphB)

        # translate to original domain
        self._reconstructA = self.g_BA(self._predictedB)
        self._reconstructB = self.g_AB(self._predictedA)

        # identity mappigs
        self._img_A_id = self.g_BA(self._xphA)
        self._img_B_id = self.g_AB(self._xphB)

        self.d_A = self.build_discriminator(df,depth)
        self.d_B = self.build_discriminator(df,depth)

        self._realA = self.d_A(self._xphA)
        self._fakeA = self.d_A(self._predictedA)
        self._realB = self.d_B(self._xphB)
        self._fakeB = self.d_B(self._predictedB)


    def build_generator(self,gf,depth):
        """U-Net Generator"""
        def conv2d(layer_input, filters, f_size=4):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            d = InstanceNormalization()(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = InstanceNormalization()(u)
            u = Concatenate()([u, skip_input])
            return u

        # Image input
        d0 = Input(shape=self.img_shape)
        input = d0
        layers = []
        # Downsampling

        for i in range(depth):
            output = conv2d(input,gf*(2**i))
            layers.append(output)
            input = output
        for i in range(depth-2, -1, -1):
            output = deconv2d(input,layers[i],gf*2**i)
            input = output

        u4 = UpSampling2D(size=2)(input)
        output_img = Conv2D(self.img_shape[-1], kernel_size=4, strides=1, padding='same', activation='sigmoid')(u4)

        return Model(d0, output_img)

    def build_discriminator(self,df,depth):
        def d_layer(layer_input, filters, f_size=4, normalization=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if normalization:
                d = InstanceNormalization()(d)
            return d

        img = Input(shape=self.img_shape)

        d1 = d_layer(img, df, normalization=False)
        input = d1
        for i in range(1,depth):
            output = d_layer(input, df*2**i)
            input = output

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(input)

        return Model(img, validity)

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



        self._genA_loss = (tf.losses.mean_squared_error(
                                predictions=self._fakeA,
                                labels=tf.ones_like(self._fakeA))
                     +      self._LAMBDA*self._cycle_loss
                     +  0.1*self._LAMBDA*tf.losses.absolute_difference(
                            predictions=self._img_A_id,
                            labels=self._xphA)
                     )

        self._genB_loss = (tf.losses.mean_squared_error(
                                predictions=self._fakeB,
                                 labels=tf.ones_like(self._fakeB))
                      + self._LAMBDA*self._cycle_loss
                      +  0.1*self._LAMBDA*tf.losses.absolute_difference(
                             predictions=self._img_B_id,
                             labels=self._xphB)
                      )

        self._reconstruction_loss = tf.losses.mean_squared_error(predictions=self._predictedB,
                                                labels=self._xphB)

    def _create_optimiser(self):
        genA_solver = tf.contrib.layers.optimize_loss(
                                        self._genA_loss,
                                        self._epoch,
                                         self.initial_learning_rate,
                                        'Adam',
                                        variables=self.g_AB.trainable_weights,
                                        increment_global_step=False,)
        genB_solver= tf.contrib.layers.optimize_loss(self._genB_loss,
                                            self._epoch,
                                             self.initial_learning_rate,
                                            'Adam',
                                            variables=self.g_BA.trainable_weights,
                                            increment_global_step=False,)
        discrimA_solver= tf.contrib.layers.optimize_loss(self._discrimA_loss,
                                                self._epoch,
                                                 self.initial_learning_rate,
                                                'Adam',
                                                variables=self.d_A.trainable_weights,
                                                increment_global_step=False,)
        discrimB_solver = tf.contrib.layers.optimize_loss(self._discrimB_loss,
                                                self._epoch,
                                                 self.initial_learning_rate,
                                                'Adam',
                                                variables=self.d_B.trainable_weights,
                                                increment_global_step=False,)
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
                        write_summary=False):
        data={self._xphA: A,
              self._xphB: B,
              K.learning_phase():True}
        summary,global_step,_,_ =self.sess.run([self._summary_op,
                                            self._batch_step,
                                            self._solver,
                                            self._batch_step_inc,],
                                            feed_dict=data)
        if write_summary:
            self._train_writer.add_summary(summary, global_step)
        return global_step

    def validate(self,A,B):
        data={self._xphA: A,
              self._xphB: B,
              K.learning_phase():False}

        summary,global_step = self.sess.run([self._validation_summary,
                                        self._batch_step],
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
        return self.sess.run(self._batch_step)

    def score(self,A,B):
        """
        The score will be determined how well A is transformed to B
        """
        data={self._xphA: A,
              self._xphB: B,
              K.learning_phase():False}
        return self.sess.run(self._reconstruction_loss,
                        feed_dict=data)
    def transform_to_A(self,B):
        data={self._xphB: B,
              K.learning_phase():False}
        return self.sess.run(self._predictedA,
                        feed_dict=data)
    def transform_to_B(self,A):
        data={self._xphA: A,
              K.learning_phase():False}
        return self.sess.run(self._predictedB,
                        feed_dict=data)
