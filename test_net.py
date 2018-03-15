import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
from tensorflow.examples.tutorials.mnist import input_data
from functools import partial

# Convolutional Layer 1.
filter_size1 = 5          # Convolution filters are 5 x 5 pixels.
num_filters1 = 16         # There are 16 of these filters.

# Convolutional Layer 2.
filter_size2 = 5          # Convolution filters are 5 x 5 pixels.
num_filters2 = 36         # There are 36 of these filters.

# convolutional layer 3.
filter_size3 = 5
num_filters3 = 36         # There are 36 of these filters.

# convolutional layer 4.
filter_size4 = 5
num_filters4 = 36         # There are 36 of these filters.

# convolutional layer 5.
filter_size5 = 5
num_filters5 = 36         # There are 36 of these filters.

# Fully-connected layer.
fc_size = 128             # Number of neurons in fully-connected laye

#data = input_data.read_data_sets('data/MNIST/', one_hot=True)

#data.test.cls = np.argmax(data.test.labels, axis=1)

train = np.load("./VOC_data/voc07_train_cropped.npy")
test = np.load("./VOC_data/voc07_test_cropped.npy")

train_labels = np.load("./VOC_data/voc07_train_labels.npy")
test_labels = np.load("./VOC_data/voc07_test_labels.npy")



# We know that MNIST images are 28 pixels in each dimension.
img_size = 50

# Images are stored in one-dimensional arrays of this length.
img_size_flat = img_size * img_size

# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)

# Number of colour channels for the images: 1 channel for gray-scale.
num_channels = 3

# Number of classes, one class for each of 10 digits.
num_classes = 20

# Get the first images from the test-set.
#images = data.test.images[0:9]

# Get the true classes for those images.
#cls_true = data.test.cls[0:9]

def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

def new_conv_layer(input,              # The previous layer.
                   num_input_channels, # Num. channels in prev. layer.
                   filter_size,        # Width and height of each filter.
                   num_filters,        # Number of filters.
                   use_pooling=True):  # Use 2x2 max-pooling.

    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create new weights aka. filters with the given shape.
    weights = new_weights(shape=shape)

    # Create new biases, one for each filter.
    biases = new_biases(length=num_filters)

    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    layer += biases

    # Use pooling to down-sample the image resolution?
    if use_pooling:
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

    layer = tf.nn.relu(layer)

    return layer, weights

def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()

    num_features = layer_shape[1:4].num_elements()

    layer_flat = tf.reshape(layer, [-1, num_features])

    return layer_flat, num_features

def new_fc_layer(input,          # The previous layer.
                 num_inputs,     # Num. inputs from prev. layer.
                 num_outputs,    # Num. outputs.
                 use_relu=True): # Use Rectified Linear Unit (ReLU)?

    # Create new weights and biases.
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    layer = tf.matmul(input, weights) + biases

    # Use ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer

# Counter for total number of iterations performed so far.
total_iterations = 0
batch_size = 64

def optimize(num_iterations, session, optimizer):
    # Ensure we update the global variable rather than a local copy.
    global total_iterations

    # Start-time used for printing time-usage below.
    start_time = time.time()

    for i in range(total_iterations,
                   total_iterations + num_iterations):

        x_batch, y_true_batch = it.get_next()

        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        session.run(optimizer, feed_dict=feed_dict_train)

        # Print status every 100 iterations.
        if i % 100 == 0:
            # Calculate the accuracy on the training-set.
            acc = session.run(accuracy, feed_dict=feed_dict_train)

            # Message for printing.
            msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"

            # Print it.
            print(msg.format(i + 1, acc))

    # Update the total number of iterations performed.
    total_iterations += num_iterations

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

x = tf.placeholder(tf.float32, shape=[None, img_size,img_size,num_channels], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, axis=1)

def main(unused_args):

    dl = partial(tf.layers.dense, activation = tf.nn.relu) # dense layer
    training = tf.placeholder_with_default(False, shape=(), name='training')
    bnl = partial(tf.layers.batch_normalization,
            training=training, momentum=0.9) # batch normalization layer

    layer_conv1, weights_conv1 = \
        new_conv_layer(input=x_image,
                    num_input_channels=num_channels,
                    filter_size=filter_size1,
                    num_filters=num_filters1,
                    use_pooling=True)
    bn1 = bnl(layer_conv1)
    bn1_act = tf.nn.elu(bn1)

    layer_conv2, weights_conv2 = \
        new_conv_layer(input=bn1_act,
                    num_input_channels=num_filters1,
                    filter_size=filter_size2,
                    num_filters=num_filters2,
                    use_pooling=True)
    bn2 = bnl(layer_conv2)
    bn2_act = tf.nn.elu(bn2)
    
    
    layer_conv3, weights_conv3 = \
            new_conv_layer(input=bn2_act,
                num_input_channels=num_filters2,
                filter_size=filter_size3,
                num_filters=num_filters3,
                use_pooling=False)
    bn3 = bnl(layer_conv3)
    bn3_act = tf.nn.elu(bn3)

    layer_conv4, weights_conv4 = \
            new_conv_layer(input=bn3_act,
                num_input_channels=num_filters3,
                filter_size=filter_size4,
                num_filters=num_filters4,
                use_pooling=False)

    layer_conv5, weights_conv5 = \
            new_conv_layer(input=layer_conv4,
                num_input_channels=num_filters4,
                filter_size=filter_size5,
                num_filters=num_filters5,
                use_pooling=True)

    layer_flat, num_features = flatten_layer(layer_conv2)

    layer_fc1 = new_fc_layer(input=layer_flat,
                            num_inputs=num_features,
                            num_outputs=fc_size,
                            use_relu=False)

    layer_fc2 = new_fc_layer(input=layer_fc1,
                            num_inputs=fc_size,
                            num_outputs=fc_size,
                            use_relu=False)

    layer_fc3 = new_fc_layer(input=layer_fc2,
                            num_inputs=fc_size,
                            num_outputs=num_classes,
                            use_relu=True)

    y_pred = tf.nn.softmax(layer_fc3)

    y_pred_cls = tf.argmax(y_pred, axis=1)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc3,
                                                            labels=y_true)


    loss = tf.reduce_mean(cross_entropy)
    #optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
    optimizer = tf.train.MomentumOptimizer(learning_rate=0.01,momentum=0.9, use_nesterov=True)
    training_op = optimizer.minimize(loss)

    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    session = tf.Session()
    session.run(tf.global_variables_initializer())

    # Ensure we update the global variable rather than a local copy.
    total_iterations = 10000
    
    # # call the img_loader
    train_dataset = tf.data.Dataset.from_tensor_slices((x,y_true)).repeat().batch(batch_size)
    #Itterator
    it = train_dataset.make_initializable_iterator()
    
    session.run(it.initializer, feed_dict={x:train, y_true: train_labels})


    # Start-time used for printing time-usage below.
    start_time = time.time()

    for i in range(total_iterations):
                   #total_iterations + num_iterations):
        print("iteration: " + str(i))

        x_batch, y_true_batch = it.get_next()

        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        session.run(training_op, feed_dict=feed_dict_train)

        # Print status every 100 iterations.
        if i % 100 == 0:
            # Calculate the accuracy on the training-set.
            acc = session.run(accuracy, feed_dict=feed_dict_train)

            # Message for printing.
            msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"

            # Print it.
            print(msg.format(i + 1, acc))

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
    session.close()


if __name__ == '__main__':
    tf.app.run(main=main)
