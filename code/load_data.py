import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from os.path import expanduser
home = expanduser("~")

from model_utils import tfrecord_utils as tfrecord
# functions to read images
from matplotlib.pyplot import imread
#from skimage.transform import resize
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf

#, load_size=140, fine_size=128, is_testing=False
def _load_train_data(image_pathA,image_pathB):
    """
    Since tensorflow cannot load jpgs (but rather jpegs)
    we are stuck using matplotlib imread functions
    """
    # load the image
    img_A = imread(image_pathA)
    img_B = imread(image_pathB)

    return img_A,img_B

def _random_resize_function(imageA_decoded,imageB_decoded,
                            load_size=140, fine_size=128,demean=True):
    """
    Takes in pair loaded images resizes them and adds some random pertubations
    """
    # load the images slightly larger than input size
    imageA_decoded.set_shape([None, None, None])
    img_A = tf.image.resize_images(imageA_decoded, [load_size, load_size])

    imageB_decoded.set_shape([None, None, None])
    img_B = tf.image.resize_images(imageB_decoded, [load_size, load_size])

    h1 = tf.random_uniform((),
                            minval=0,
                            maxval=load_size-fine_size,
                            dtype=tf.int32)
    w1 =  tf.random_uniform((),
                            minval=0,
                            maxval=load_size-fine_size,
                            dtype=tf.int32)
    # randomly crop images to finesize
    img_A = img_A[h1:h1+fine_size, w1:w1+fine_size]
    img_B = img_B[h1:h1+fine_size, w1:w1+fine_size]

    if demean:
        img_A = img_A/127.5 - 1.0
        img_B = img_B/127.5 - 1.0
    return img_A,img_B

def _random_flip(imageA,imageB):
    prob = tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
    predicate = tf.less(prob, 0.5)

    img_A = tf.cond(predicate,lambda:tf.image.flip_left_right(imageA),lambda: imageA)
    img_B = tf.cond(predicate,lambda:tf.image.flip_left_right(imageB),lambda: imageB)

    return img_A, img_B


def _resize_function(imageA_decoded,imageB_decoded,fine_size=128):
    """
    Takes in pair loaded images resizes them
    """
    imageA_decoded.set_shape([None, None, None])
    img_A = tf.image.resize_images(imageA_decoded, [fine_size, fine_size])

    imageB_decoded.set_shape([None, None, None])
    img_B = tf.image.resize_images(imageB_decoded, [fine_size, fine_size])

    img_A = img_A/127.5 - 1.0
    img_B = img_B/127.5 - 1.0
    return img_A,img_B

def _brats_parser(fileA,fileB,feature_size=128):
    features_dict={'data':tf.FixedLenFeature([feature_size**2], tf.float32)}
    imgA = tf.parse_single_example(fileA,
                                    features=features_dict)
    imgB = tf.parse_single_example(fileB,
                                    features=features_dict)
    return tf.reshape(imgA['data'],[feature_size,feature_size,1]),tf.reshape(imgB['data'],[feature_size,feature_size,1])

class DataQueue(object):
    def __init__(self,dir_A,dir_B,n_epochs=None,is_validation=False,queue_size=30,
                     separate_shuffle=False):
        self.dir_A = dir_A
        self.dir_B = dir_B
        self.filenames_A = tf.placeholder(tf.string, shape=[None])
        self.filenames_B = tf.placeholder(tf.string, shape=[None])
        self.batch_ph = tf.placeholder(tf.int64,shape=())



        datasetA = tf.data.TFRecordDataset((self.filenames_A))
        datasetB = tf.data.TFRecordDataset((self.filenames_B))

        if separate_shuffle:
            datasetA = datasetA.shuffle(buffer_size=self.batch_ph*queue_size)
            datasetB = datasetA.shuffle(buffer_size=self.batch_ph*queue_size)
        dataset = datasetA.zip((datasetA,datasetB))
        feature_size=128
        dataset = dataset.map(lambda x,y: _brats_parser(x,y,feature_size))

        if not is_validation:
            dataset = dataset.map(lambda x,y:
                _random_resize_function(x,y,demean=False))

        dataset = dataset.repeat(n_epochs)

        # shuffler
        dataset = dataset.shuffle(buffer_size=self.batch_ph*queue_size)

        dataset = dataset.batch(tf.cast(self.batch_ph,tf.int64))
        self.iterator = dataset.make_initializable_iterator()
        self.iterator_next = self.iterator.get_next()

    def initialise_queue(self,batch_size,sess):
        #if shuffle_files:
            # shuffle the files separately
        #    np.random.shuffle(self.dir_A)
        #    np.random.shuffle(self.dir_B)

        self.sess = sess
        self.sess.run(self.iterator.initializer,
                      feed_dict={self.filenames_A:self.dir_A,
                                self.filenames_B:self.dir_B,
                                self.batch_ph: batch_size})
    def get_next(self):
        return self.sess.run(self.iterator_next)

def load_data(A_train_dir,B_train_dir,
            A_test_dir, B_test_dir,is_semi_supervised=True,
            paired_percent=0.1,paired_train_dir=None):

    test_files_A = np.sort(tfrecord.get_files_in_dir([A_test_dir]))

    test_files_B = np.sort(tfrecord.get_files_in_dir([B_test_dir]))


    if is_semi_supervised:
        if paired_train_dir is not None:

            A_pair = np.sort(tfrecord.get_files_in_dir([paired_train_dir[0]]))
            B_pair = np.sort(tfrecord.get_files_in_dir([paired_train_dir[1]]))

            A_unpair = np.sort(tfrecord.get_files_in_dir([A_train_dir,paired_train_dir[0]]))
            B_unpair = np.sort(tfrecord.get_files_in_dir([B_train_dir,paired_train_dir[1]]))

        else:
            A_unpair,A_pair,B_unpair,B_pair = train_test_split(training_files_A,
                            training_files_B,
                            random_state=42,
                            test_size=paired_percent,
                            shuffle=True)
    else:
        if paired_train_dir is not None:
            A_unpair = np.sort(tfrecord.get_files_in_dir([A_train_dir,paired_train_dir[0]]))
            B_unpair = np.sort(tfrecord.get_files_in_dir([B_train_dir,paired_train_dir[1]]))
        else:
            A_unpair = np.sort(tfrecord.get_files_in_dir([A_train_dir]))
            B_unpair = np.sort(tfrecord.get_files_in_dir([B_train_dir]))

    if len(A_unpair) == 0:
        raise ValueError("empty directory")
    else:
        if is_semi_supervised:
            print("Working with %d unpaired files and %d paired files" %(len(A_unpair),len(A_pair)))
        else:
            print("Working with %d unpaired files and 0 paired files" %len(A_unpair))
    data_queue = {}

    # create the queue
    unpaired_queue = DataQueue(A_unpair,B_unpair,n_epochs=1,separate_shuffle=True)


    validation_queue = DataQueue(test_files_A,test_files_B,n_epochs=None,
                                is_validation=True,queue_size=5)

    data_queue['unpaired'] = unpaired_queue
    data_queue['validation'] = validation_queue

    if is_semi_supervised:
        paired_queue = DataQueue(A_pair,B_pair,n_epochs=None)
        data_queue['paired'] = paired_queue


    return data_queue
