import tensorflow as tf
import numpy as np
import time
from datetime import timedelta
from tensorflow.examples.tutorials.mnist import input_data
from functools import partial
import os

# Convolutional Layer 1
filter_size1 = 5
num_filters1 = 96

# Convolutional Layer 2
filter_size2 = 5
num_filters2 = 256

# convolutional layer 3
filter_size3 = 5
num_filters3 = 384

# convolutional layer 4
filter_size4 = 5
num_filters4 = 384

# convolutional layer 5
filter_size5 = 5
num_filters5 = 256

# Fully-connected layer
fc_size = 1000


train = np.load("./VOC_data/voc07_train_only_blurred_nn_cropped.npy")
# cv = np.load("./VOC_data/voc07_cv_only_blurred_nn_cropped.npy")

train_labels = np.load("./VOC_data/voc07_train_only_labels.npy")
# test_labels = np.load("./VOC_data/voc07_test_labels3.npy")

img_size = 227

num_channels = 3

# Number of classes, one class for each of 10 digits.
num_classes = 20


def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=np.sqrt(2) * np.sqrt(2.0 / (shape[2]+shape[3]))))

def new_biases(length):
    return tf.Variable(tf.constant(0.5, shape=[length]))

def new_conv_layer(input,              # The previous layer
                   num_input_channels, # Num. channels in prev. layer
                   filter_size,        # Width and height of each filter
                   num_filters,        # Number of filters
                   use_pooling=True,   # Use 2x2 max-pooling
                   padding=['SAME', 'VALID'], # Type of padding for conv/pooling respectively
                   use_relu=False, # Should we ReLU after convolution layer
                   use_local_norm=False): # should we use local resp. norm. for convolution layer

    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create new weights aka. filters with the given shape
    weights = new_weights(shape=shape)

    # Create new biases, one for each filter
    biases = new_biases(length=num_filters)

    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding=padding[0])
    layer += biases

    if use_local_norm:
        layer = tf.nn.local_response_normalization(layer,
        depth_radius=2,
        bias=1,
        alpha=0.00002,
        beta=0.75)

    if use_relu:
        layer = tf.nn.relu(layer)

    if use_pooling:
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding=padding[1])

    return layer, weights

def flatten_layer(layer):
    # Get the shape of the input layer
    layer_shape = layer.get_shape()

    num_features = layer_shape[1:4].num_elements()

    layer_flat = tf.reshape(layer, [-1, num_features])

    return layer_flat, num_features

batch_size = 50

x = tf.placeholder(tf.float32, shape=[None, img_size,img_size,num_channels], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
y0 = tf.placeholder(tf.int64, shape=(None), name='y0')

def l_relu(z, name=None):
    return tf.maximum(0.01 * z, z, name=name)

def run_net(y_labs, y_true):
    # He initialization for weights to help avoid vanishing/exploding
    he_init = tf.contrib.layers.variance_scaling_initializer(factor=1, mode='FAN_AVG', uniform=False)
    # Dense layer
    dl = partial(tf.layers.dense, activation = tf.nn.relu, kernel_initializer=he_init, use_bias=True, name=None)
    training = tf.placeholder_with_default(True, shape=(), name='training')
    # Batch normalization layer
    bnl = partial(tf.layers.batch_normalization,
            training=training, momentum=0.99, center=True, scale=True)

    layer_conv1, weights_conv1 = \
        new_conv_layer(input=x_image,
                    num_input_channels=num_channels,
                    filter_size=filter_size1,
                    num_filters=num_filters1,
                    use_pooling=True,
                    use_relu=True,
                    use_local_norm=True)
    bn1 = bnl(inputs=layer_conv1)
    bn1_act = tf.nn.elu(bn1)

    layer_conv2, weights_conv2 = \
        new_conv_layer(input=bn1_act,
                    num_input_channels=num_filters1,
                    filter_size=filter_size2,
                    num_filters=num_filters2,
                    use_pooling=True,
                    use_relu=True,
                    use_local_norm=True)
    bn2 = bnl(inputs=layer_conv2)
    bn2_act = tf.nn.elu(bn2)

    layer_conv3, weights_conv3 = \
            new_conv_layer(input=bn2_act,
                num_input_channels=num_filters2,
                filter_size=filter_size3,
                num_filters=num_filters3,
                use_pooling=False,
                use_relu=True,
                use_local_norm=False)

    layer_conv4, weights_conv4 = \
            new_conv_layer(input=layer_conv3,
                num_input_channels=num_filters3,
                filter_size=filter_size4,
                num_filters=num_filters4,
                use_pooling=False,
                use_relu=True,
                use_local_norm=False)

    layer_conv5, weights_conv5 = \
            new_conv_layer(input=layer_conv4,
                num_input_channels=num_filters4,
                filter_size=filter_size5,
                num_filters=num_filters5,
                use_pooling=True,
                use_relu=True,
                use_local_norm=False)



    layer_flat, num_features = flatten_layer(layer_conv5)

    layer_fc1 = dl(layer_flat, fc_size, activation=tf.nn.relu, name="fc1")
    bn6 = bnl(layer_fc1)
    # bn6_act = tf.nn.elu(bn6)
    weights6 = tf.get_default_graph().get_tensor_by_name(
        os.path.split(layer_fc1.name)[0] + '/kernel:0')

    layer_fc2 = dl(bn6, fc_size, activation=tf.nn.relu, name="fc2")
    bn7 = bnl(layer_fc2)
    # bn7_act = tf.nn.elu(bn7)
    weights7 = tf.get_default_graph().get_tensor_by_name(
        os.path.split(layer_fc2.name)[0] + '/kernel:0')

    outputs = dl(bn7, num_classes, activation=tf.nn.relu, name="outputs")
    pre_logits = bnl(outputs)
    logits = tf.nn.relu(pre_logits)
    weights8 = tf.get_default_graph().get_tensor_by_name(
        os.path.split(outputs.name)[0] + '/kernel:0')

    logits = tf.clip_by_value(logits, 0, 1)

    # Cross entropy cost function
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,
                                                            labels=tf.reshape(y_true, [batch_size, num_classes]))
    loss = tf.reduce_mean(cross_entropy)
    # We will be using momemtum descent with nesterov optimizationaa
    optimizer = tf.train.MomentumOptimizer(learning_rate=0.01,momentum=0.9, use_nesterov=True)
    # Operations needed to run every iteration
    training_op = optimizer.minimize(loss)
    # Needed to deal with batch normalization operations
    extra_update = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    # Calculate accuracy; since logits is a float, it will round to 0 or 1 and then cast to int
    # for comparisons
    correct_prediction = tf.equal(tf.cast(tf.round(logits), tf.int64), y_true)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    session = tf.Session()
    init = tf.global_variables_initializer()
    session.run(init)

    # Ensure we update the global variable rather than a local copy.
    total_iterations = 101

    # # call the img_loader
    train_dataset = tf.data.Dataset.from_tensor_slices((x,y_true)).repeat().batch(batch_size)
    #Iterator
    it = train_dataset.make_initializable_iterator()

    # Object to save our model after training
    saver = tf.train.Saver()
    session.run(it.initializer, feed_dict={x:train, y_true: y_labs})

    start_time = time.time()
    # Iteratior object to get every batch in for loop
    x_batch, y_true_batch = it.get_next()

    for i in range(total_iterations):
                   #total_iterations + num_iterations):
        print("iteration: " + str(i))

        X_eval, y_eval = session.run([x_batch, y_true_batch])

        feed_dict_train = {x: X_eval,
                           y_true: y_eval}


        session.run([training_op, extra_update], feed_dict=feed_dict_train)

        # Print status every 5 iterations
        if i % 5 == 0:
            # Calculate the accuracy on the training-set
            acc = session.run(accuracy, feed_dict=feed_dict_train)

            # Message for printing.
            msg = "Optimization Iteration: " +str(i+1)+", Training Accuracy: " + str(acc)
            print(msg)
            print("Checkpoint..")
            save_path = saver.save(session, "./temp_voc07_model.ckpt")

    # Ending time
    end_time = time.time()
    save_path = saver.save(session, "./voc07_model.ckpt")
    # Difference between start and end-times
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
    session.close()


def main(unused_arg):
    c0 = train_labels
    # c1 = train_labels[:,1]
    # c2 = train_labels[:,2]
    # c3 = train_labels[:,3]
    # c4 = train_labels[:,4]
    # c5 = train_labels[:,5]
    # c6 = train_labels[:,6]
    # c7 = train_labels[:,7]
    # c8 = train_labels[:,8]
    # c9 = train_labels[:,9]
    # c10 = train_labels[:,10]
    # c11 = train_labels[:,11]
    # c12 = train_labels[:,12]
    # c13 = train_labels[:,13]
    # c14 = train_labels[:,14]
    # c15 = train_labels[:,15]
    # c16 = train_labels[:,16]
    # c17 = train_labels[:,17]
    # c18 = train_labels[:,18]
    # c19 = train_labels[:,19]
    # w0 = run_net(np.transpose(np.array([c0])), y0)
    w0 = run_net(c0, y0)
    # w1 = run_net(c1, y1)
    # w2 = run_net(c2, y2)
    # w3 = run_net(c3, y3)
    # w4 = run_net(c4, y4)
    # w5 = run_net(c5, y5)
    # w6 = run_net(c6, y6)
    # w7 = run_net(c7, y7)
    # w8 = run_net(c8, y8)
    # w9 = run_net(c9, y9)
    # w10 = run_net(c10, y19)
    # w11 = run_net(c11, y11)
    # w12 = run_net(c12, y12)
    # w13 = run_net(c13, y13)
    # w14 = run_net(c14, y14)
    # w15 = run_net(c15, y15)
    # w16 = run_net(c16, y16)
    # w17 = run_net(c17, y17)
    # w18 = run_net(c18, y18)
    # w19 = run_net(c19, y19)

if __name__ == '__main__':
    tf.app.run(main=main)
