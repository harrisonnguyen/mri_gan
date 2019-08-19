import tensorflow as tf

def linear_decay(learning_rate,global_step,
                begin_decay,**kwargs):
    step = tf.math.maximum(global_step-begin_decay,0)
    return tf.train.polynomial_decay(learning_rate,step,**kwargs)

def main():
    epoch = tf.Variable(0,trainable=False,dtype=tf.int32)
    lr = linear_decay(1.0,epoch,begin_decay=10,decay_steps=100)
    epoch_inc = tf.assign_add(epoch,1)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        i,e =sess.run([lr,epoch_inc])
        print("Epoch {}: learning rate:{}".format(e,i))


if __name__ == "__main__":
    main()
