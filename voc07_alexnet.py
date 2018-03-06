import tensorflow as tf
import numpy as np
from functools import partial
from PIL import Image

# a function to create convolutional layers
def new_conv_layer(input,              # The previous layer.
                   num_input_channels, # Num. channels in prev. layer.
                   filter_size,        # Width and height of each filter.
                   num_filters,        # Number of filters.
                   use_pooling=True):  # Use 2x2 max-pooling.

    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create new weights aka. filters with the given shape.
    weights = tf.Variable(tf.truncated_normal(shape, stddev=0.05))

    # Create new biases, one for each filter.
    biases = tf.Variable(tf.constant(0.05, shape=[num_filters]))

    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    layer += biases

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

n_inputs = 500*500 #mnist
img_size = 500
num_channels = 3
num_examples = 5011


######################################################################################################
# constants for height and width to resize all pics to
width = 500 # width
height = 500 # height 
ch = 3 #number fo channels in this image set, 3 for RGB
voc_num_train_imgs = 5011 # numeber of images in directory
#VOC data location
voc_train_location = "./datasets/VOC_2007/traindata/VOC2007/JPEGImages/"
voc_img_type = "jpg"

# call the img_loader
# #########################################################################################################
voc_num_test_imgs = 4952
voc_test_location = "./datasets/VOC_2007/testdata/VOC2007/JPEGImages/"



# imgs = np.load(".Object-Local-Deep-Learning/VOC_data/voc07_train.npy")
# training_imgs = imgs['arr_1']
# testing_imgs = imgs['arr_0']
#########################################################################################################
#trainlabels, testlabels = get_labels()
#########################################################################################################


# conv lay 1
filter_size1 = 5
num_filters1 = 16

#conv lay 2
filter_size2 = 5
num_filters2 = 36

#conv lay 3
filter_size3 = 3
num_filters3 = 128

#conv lay 4
filter_size4 = 3
num_filters4 = 256

#conv lay 5
filter_size5 = 3
num_filters5 = 512


#full conn lay
fc_size1 = 1000

n_hidden1 = 300
n_hidden2 = 300
n_outputs = 20
learning_rate = 0.01 # grad descent param
n_epochs = 40
batch_size = 50
scale = 0.0004 # l1 regularization param
dropout_rate = 0.5

def main(unused_arg):

    train_data = np.load("./VOC_data/voc07_train_padded.npy")
    # test_data = np.load("./VOC_data/voc07_test_padded.npy")
    train_labels = np.load("./VOC_data/voc07_train_labels.npy")
    # test_labels = np.load("./VOC_data/voc07_test_labels.npy")

    

    X = tf.placeholder(tf.float32, shape=(None, 500, 500, 3), name = "X")
    X_rs = tf.reshape(X, [-1, img_size, img_size, num_channels])
    y = tf.placeholder(tf.float32, shape=(None,20), name="y")
    # # call the img_loader
    train_dataset = tf.data.Dataset.from_tensor_slices((X,y)).repeat().batch(batch_size)
    #bt = train_dataset.batch(50)

    it = train_dataset.make_initializable_iterator()



    dl = partial(tf.layers.dense, activation = tf.nn.relu) # dense layer

    # set up graph
    with tf.name_scope("dnn"):
        training = tf.placeholder_with_default(False, shape=(), name='training')
        bnl = partial(tf.layers.batch_normalization,
            training=training, momentum=0.9) # batch normalization layer

        h1, weights_conv1 = \
            new_conv_layer(input=X_rs,
                num_input_channels=num_channels,
                filter_size=filter_size1,
                num_filters=num_filters1,
                use_pooling=True)
        bn1 = bnl(h1)
        bn1_act = tf.nn.elu(bn1)

        h2, weights_conv2 = \
            new_conv_layer(input=bn1_act,
                num_input_channels=num_filters1,
                filter_size=filter_size2,
                num_filters=num_filters2,
                use_pooling=True)
        bn2 = bnl(h2)
        bn2_act = tf.nn.elu(bn2)

        h3, weights_conv3 = \
            new_conv_layer(input=bn2_act,
                num_input_channels=num_filters2,
                filter_size=filter_size3,
                num_filters=num_filters3,
                use_pooling=False)
        bn3 = bnl(h3)
        bn3_act = tf.nn.elu(bn3)

        h4, weights_conv4 = \
            new_conv_layer(input=bn3_act,
                num_input_channels=num_filters3,
                filter_size=filter_size4,
                num_filters=num_filters4,
                use_pooling=False)

        h5, weights_conv5 = \
            new_conv_layer(input=h4,
                num_input_channels=num_filters4,
                filter_size=filter_size5,
                num_filters=num_filters5,
                use_pooling=True)
        
        layer_flat, num_features = flatten_layer(h5)

        fc1 = dl(layer_flat, fc_size1)
        fc2 = dl(fc1, fc_size1)

        logits = dl(fc2, n_outputs, activation=None,name="outputs")

    # get loss function
    with tf.name_scope("loss"):
        xentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y,
                                                                logits=logits)
        loss = tf.reduce_mean(xentropy, name="loss")

    # using moment descent with nesterov
    with tf.name_scope("train"):
        optimizer = tf.train.MomentumOptimizer(learning_rate,momentum=0.9, use_nesterov=True)
        training_op = optimizer.minimize(loss)

    #with tf.name_scope("eval"):
        # correct = tf.nn.in_top_k(logits, y, 1)
        # accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    init = tf.global_variables_initializer()
    # saver = tf.train.Saver()
    print("starting sess")
    with tf.Session() as sess:
        init.run()
        print("started")
        #save_path = saver.save(sess, "./VOC_data/voc_model.ckpt")
        print("Model saved in path")
        # Restore variables from disk.
        # saver.restore(sess, "/tmp/model.ckpt")
        # print("Model restored.")

        sess.run(it.initializer, feed_dict={X:train_data, y: train_labels})

        for epoch in range(n_epochs):
            for iteration in range(num_examples // batch_size):

                X_batch, y_batch = it.get_next()

                #training
                sess.run(training_op,
                        feed_dict={X: X_batch, y: y_batch})
            acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
            #acc_test = accuracy.eval(feed_dict={X: testing_img,
            #                                    y: testlabels})
        #     print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)
        # save_path = saver.save(sess, "./my_model_final.ckpt")
            print(epoch, "Train accuracy:", acc_train)
        save_path = saver.save(sess, "./my_model_final.ckpt")


if __name__ == '__main__':
    tf.app.run(main=main)
