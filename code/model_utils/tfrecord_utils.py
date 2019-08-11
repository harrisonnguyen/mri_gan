import numpy as np # linear algebra
import tensorflow as tf
import os


"""
methods to write to tfrecord
"""
def write_tfrecord(file_name, data_array,feature_list,data_list,directory):
    instances = data_array[0].shape[0]
    #str_num = str(n).zfill(2)
    #str_num = str(n)
    # write the data
    if not os.path.exists(directory):
        os.makedirs(directory)
    writer = tf.python_io.TFRecordWriter(os.path.join(directory,file_name +".tfrecords"))

        #iterate over each example
    for i in range(instances):
        temp_dict = {}
        for j in range(0,len(feature_list)):
            if data_list[j] == 'float':
                temp_dict[feature_list[j]] = tf.train.Feature(float_list=tf.train.FloatList(value = data_array[j][i].astype(float)))
            elif data_list[j] == 'int':
                temp_dict[feature_list[j]] = tf.train.Feature(int64_list=tf.train.Int64List(value = [data_array[j][i]]))
            elif data_list[j] =='str':
                temp_dict[feature_list[j]] = tf.train.Feature(bytes_list=tf.train.BytesList(value = [data_array[j][i]]))

        #construct the example proto object
        example = tf.train.Example(
                    features = tf.train.Features(
                        feature = temp_dict))
        # use the proto object to serialize the example to a string
        serialized = example.SerializeToString()
        # write the serialized object to disk
        writer.write(serialized)
    #writer.close()


def tfrecord_parser(serialized_example,feature_list,feature_type,feature_size):
    """Parses a single tf.Example into image and label tensors.
       e.g feature_list = ['mfcc','melspec','length','label','id']
            feature_type = ['float','float','int','int','str']
            serialized_example = some_file.tfrecord

    """
    # One needs to describe the format of the objects to be returned

    features_dict = {}
    for i in range(len(feature_list)):
        if feature_type[i] == 'float':
            features_dict[feature_list[i]] = tf.FixedLenFeature([feature_size], tf.float32)
        elif feature_type[i] == 'int':
            features_dict[feature_list[i]] = tf.FixedLenFeature([], tf.int64)
        elif feature_type[i] =='str':
            features_dict[feature_list[i]] = tf.FixedLenFeature([],tf.string)
        else:
            raise ValueError('Not Valid data type')
    features = tf.parse_single_example(
                                        serialized_example,
                                        features=features_dict)
    return  [features[ele] for ele in feature_list]

def create_tfrecord_queue(parser_fn,n_epochs=None,filename_ph=None,batch_ph=None):
    # filenames ffor validation/training
    if filename_ph is None:
        filename_ph = tf.placeholder(tf.string, shape=[None])
    if batch_ph is None:
        batch_ph = tf.placeholder(tf.int64,shape=())
    dataset = tf.data.TFRecordDataset(filename_ph)

    if n_epochs is None:
        # Repeat the input indefinitely.
        dataset = dataset.repeat()
    else:
        dataset = dataset.repeat(n_epochs)

    #convert byte string to something meaningful
    # some parser functions thaty uses lambda x: parser_fn(x,...,,)
    dataset = dataset.map(parser_fn)

    # shuffler
    dataset = dataset.shuffle(buffer_size=batch_ph*5)

    dataset = dataset.batch(tf.cast(batch_ph,tf.int64))
    iterator = dataset.make_initializable_iterator()
    iterator_next = iterator.get_next()

    return iterator,iterator_next,filename_ph, batch_ph,

def get_files_in_dir(directories,file_format=None):
    """
    gets all files in all subolfders in the for the given directories

    Args:
        directories: an array of strings (directory names)
        file_format: a string showing the fileformat to search for (e.g. ".jpg",".nii")

    Returns:
        A list of strings of files
    """
    files_names = []
    for root in directories:
        for path, subdirs, files in os.walk(root):
            for name in files:
                if file_format is not None:
                    if file_format in name:
                        files_names.append(os.path.join(path, name))
                else:
                    files_names.append(os.path.join(path, name))
    return files_names
