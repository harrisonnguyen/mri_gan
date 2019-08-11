import tensorflow as tf



def get_params(layers):
    """ Get trainable parameters of specified layer."""
    params=[]
    for ele in layers:
        variables = tf.trainable_variables()
        params.append([v for v in variables if v.name.startswith(ele+'/')])

    return params

def polynomial_decay(a,b,decay_steps,end_learning_rate=2e-7,power=1.0):
    return tf.train.polynomial_decay(
            learning_rate=a,
            global_step=b,
            decay_steps=decay_steps,
            end_learning_rate=end_learning_rate,
            power=power
        )


def lstm(network_input,state_ph,dropout_ph,early_stop_ph,
          n_layers,state_size,n_labels,cell_type='gru'):
    # we produce the state of each layer into an element of an array
    state_per_layer_list = tf.unstack(state_ph, axis=0)
    if cell_type =='gru':
        rnn_tuple_state = tuple([state_per_layer_list[idx] for idx in range(n_layers)])
        tf_cell = tf.nn.rnn_cell.GRUCell
    elif cell_type =='lstm':
        rnn_tuple_state = tuple(
            [tf.nn.rnn_cell.LSTMStateTuple(
                state_per_layer_list[idx][0],
                state_per_layer_list[idx][1])
             for idx in range(n_layers)]
        )
        tf_cell = tf.nn.rnn_cell.LSTMCell


    #state_array = [int(state_size/i) for i in range(1,n_layers+1)]
    state_array = [state_size]*n_layers
    # forward pass
    single_cell = [tf.nn.rnn_cell.DropoutWrapper(tf_cell(size),output_keep_prob=dropout_ph)
                   for size in state_array]
    # combine cells
    cell = tf.nn.rnn_cell.MultiRNNCell(single_cell,
                                       state_is_tuple=True)

    states_series, current_state = tf.nn.dynamic_rnn(cell,
                                                     network_input,
                                                     dtype=tf.float32,
                                                     initial_state=rnn_tuple_state,
                                                     sequence_length=early_stop_ph)





    #  state_series is of shape (batch_size,sequence,hidden_state)
    # we create a set of indices to extract the matrix
    # where the second column represents the end
    # i.e. [[0, early_stop_ph[0]],[1,early_stop_ph[1]],[2, early_stop_ph[3]]...]
    index = tf.concat([tf.reshape(tf.range(0, tf.shape(early_stop_ph)[0], 1),[-1,1]),
                   tf.reshape(early_stop_ph-1,[-1,1])],axis=1)
    last = tf.gather_nd(states_series,index)
    # the transform for the output
    logits = tf.layers.dense(inputs=last,
                         units=n_labels,
                         activation=None,
                         use_bias=True)

    return logits, current_state

def create_summary(variables,types,names):
    """
    variables: a list of tensor variables
    types: a list strings of either 'scalar','image','histogram' of same length as variables
    names: a list of strings for the names of each summary
    """
    for i in range(len(variables)):
        if types[i] == 'scalar':
            tf.summary.scalar(names[i], variables[i])
        elif types[i] == 'image':
            tf.summary.image(names[i], variables[i],max_outputs=2)
        elif types[i] == 'histogram':
            tf.summary.histogram(names[i], variables[i], collections=['weights'])
        else:
            raise ValueError("Not valid summary type")
    summary_op = tf.summary.merge_all()
    weight_op = tf.summary.merge_all(key='weights')

    return summary_op,weight_op

def create_solver(loss,global_step,learning_rate_ph,decay_step_ph,params,increment_global_step,optimiser='Adam',**kwargs):
    #global_step = tf.Variable(0,trainable=False,dtype=tf.int32)
    solver = tf.contrib.layers.optimize_loss(loss,
                                         global_step=global_step,
                                         learning_rate=learning_rate_ph,
                                         learning_rate_decay_fn =
                                             lambda a,b: polynomial_decay(
                                              a,b,decay_steps=decay_step_ph,**kwargs),
                                         optimizer=optimiser,
                                         variables=params,
                                         summaries=["gradients"],
                                         increment_global_step=increment_global_step)
    return solver #, global_step
