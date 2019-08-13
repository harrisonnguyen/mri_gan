import pandas as pd
import tensorflow as tf

def _parse_image_function(example_proto,image_size):
    image_feature_description = {
    'img': tf.io.FixedLenFeature(image_size, tf.float32),
    }
    # Parse the input tf.Example proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, image_feature_description)

def extract_patch(image,*args,**kwargs):
    patches= tf.extract_volume_patches(
                image,
                ksizes=kwargs['ksizes'],
                strides=kwargs['strides'],
               padding="SAME",)
    return patches

def extract_patches_inverse(y,image_size,*args,**kwargs):
    # x is the original size of the image
    _x = tf.zeros(image_size)
    _y = extract_patch(_x,*args,**kwargs)
    #y = tf.reshape(y,tf.shape(_y))
    grad = tf.gradients(_y, _x)[0]
    # Divide by grad, to "average" together the overlapping patches
    # otherwise they would simply sum up
    return tf.gradients(_y, _x, grad_ys=y)[0] / grad

def augment(image,ksizes,strides,is_3D):
    image = image['img']
    if is_3D:
        patches = extract_patch(image,
                                ksizes=ksizes,
                                strides=strides)
        image = tf.reshape(patches,[-1,ksizes[1],ksizes[2],ksizes[3],1])

    else:
        paddings = tf.constant([[0,0,],[0,0,],[3,4],[0,0]])
        image = tf.pad(image,paddings,"CONSTANT")
        # getting the saggital slices
        #image = tf.transpose(image,[2,0,1,3])
        # crop to 128/128
        image = tf.image.crop_to_bounding_box(
                image,
                offset_height=8,
                offset_width=0,
                target_height=128,
                target_width=128
            )
        # remove any slices that contain 0
        image = remove_zeros(image)
    image = image/tf.reduce_max(image)
    image = 2.0*image-1.0
    return image

def remove_zeros(x):
    intermediate_tensor = tf.reduce_sum(tf.abs(x), [1,2,3])
    bool_mask = tf.not_equal(intermediate_tensor, 0.0)
    omit_zeros = tf.boolean_mask(x, bool_mask,axis=0)
    return omit_zeros

def load_data(filenames,
                image_size,
                ksizes=None,
                strides=None,
                is_3D=False,
                buffer_size=30):
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(lambda x:_parse_image_function(x,image_size))
    dataset = dataset.map(lambda x:augment(x,ksizes,strides,is_3D))
    dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset= dataset.repeat(1)
    return dataset


def main():


    df = pd.read_csv("data/demographics.csv")
    df["Filename"] = df["Filename"].str.replace('.nii',".tfrecords")
    df["Filename"] = "data/mri_full/" + df["Filename"].astype(str)
    image_size = [1,121,145,121,1]
    #dataset = load_data(df["Filename"].values[:5],
    #                    image_size=[1,121,145,121,1],
    #                    batch_size=1,
    #                    ksizes=[1,64,64,64,1],
    #                    strides=[1,32,32,32,1],
    #                    is_3D=True)
    dataset = load_data(df["Filename"].values[:5],
                        image_size=[121,145,121,1],
                        is_3D=False)
    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
    next_image = iterator.get_next()
    sess = tf.compat.v1.Session()
    while True:
        try:
            image = sess.run(next_image)
            print(image.shape)
            #original_image = sess.run(extract_patches_inverse(
            #                        next_image,
            #                        image_size,
            #                        ksizes=[1,64,64,64,1],
            #                        strides=[1,32,32,32,1]))
        except tf.errors.OutOfRangeError:
            break

if __name__ == "__main__":
    main()
