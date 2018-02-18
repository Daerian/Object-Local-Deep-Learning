import tensorflow as tf
import numpy as np
from functools import partial
from PIL import Image


def get_labels():
    filename_queue = tf.train.string_input_producer([".datasets/VOC_2007/traindata/VOC2007/ImageSets/Main/trainval_labels.csv", "datasets./VOC_2007/testdata/VOC2007/ImageSets/Main/test_labels.csv"])
    reader = tf.TextLineReader()
    train_key, train_vals = reader.read(filename_queue)
    test_key, test_vals = reader.read(filename_queue)

    # Default values, in case of empty columns. Also specifies the type of the
    # decoded result.
    record_defaults = [[1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1]]
    name,aeroplane,bicycle,bird,boat,bottle,bus,car,cat,chair,cow,diningtable,dog,horse,motorbike,person,pottedplant,sheep,sofa,train,tvmonitor = tf.decode_csv(train_vals, record_defaults=record_defaults)
    multi_trainlabels = tf.stack([aeroplane,bicycle,bird,boat,bottle,bus,car,cat,chair,cow,diningtable,dog,horse,motorbike,person,pottedplant,sheep,sofa,train,tvmonitor])

    test_name,test_aeroplane,test_bicycle,test_bird,test_boat,test_bottle,test_bus,test_car,test_cat,test_chair,test_cow,test_diningtable,test_dog,test_horse,test_motorbike,test_person,test_pottedplant,test_sheep,test_sofa,test_train,test_tvmonitor = tf.decode_csv(test_vals, record_defaults=record_defaults)
    multi_testlabels = tf.stack([test_aeroplane,test_bicycle,test_bird,test_boat,test_bottle,test_bus,test_car,test_cat,test_chair,test_cow,test_diningtable,test_dog,test_horse,test_motorbike,test_person,test_pottedplant,test_sheep,test_sofa,test_train,test_tvmonitor])


    with tf.Session() as sess:
    # Start populating the filename queue.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        trainlabels = np.array([])
        testlabels = np.array([])
        for i in range(5011):
            # Retrieve a single instance:
            # print("GDAKMGDAKLGDKLFGKLFGALAFGJL " + str(i))
            example = sess.run([multi_trainlabels])
            np.append(trainlabels, example)
            # print(example)
            # print(type(multi_labels))
        
        for i in range(4952):
            example = sess.run([multi_testlabels])
            np.append(testlabels, example)


    coord.request_stop()
    coord.join(threads)
    return (trainlabels, testlabels)

# a function to load images froma folder into a tensorflow dataset
def load_imgs(location, img_type, num_imgs, width, height,num_channels):
  location = location + "/*." + img_type
  print("reading from: " + location)

  #The queue for all the files in the firectory
  voc_queue = tf.train.string_input_producer(
  tf.train.match_filenames_once(location))


  #IMG reader, stores in string format
  image_reader = tf.WholeFileReader()

  #image tensor for read images above
  img_tensor = []

  print("decoding images ...")

  # begin decoding
  for i in range(num_imgs):
    #Reads the file at the top of the queue, first arg is name, irrellevant
    _, image_file = image_reader.read(voc_queue)

    #Decoder that turns the above img reader string into a tensor
    voc_image = tf.image.decode_jpeg(
      contents = image_file,
      channels = num_channels)

    #add to list of tensors
    img_tensor.append(voc_image)
    
  
  print("reshaping images ...")
  #Reshape to the sppecifications wanted, wither cropping or padding
  for img in img_tensor:
    img = tf.image.resize_image_with_crop_or_pad(
      image = img, # current image
      target_height = height, #given width
      target_width = width #given height
    )
    # print("next imaage reshaped to: " + img.get_shape())
  
  return (img_tensor)


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
training_imgs = load_imgs(
location = voc_train_location,
img_type = voc_img_type,
num_imgs = voc_num_train_imgs, 
width = width, 
height = height, 
num_channels = num_channels)
# #########################################################################################################
voc_num_test_imgs = 4952
voc_test_location = "./datasets/VOC_2007/testdata/VOC2007/JPEGImages/"

# # call the img_loader
testing_img = load_imgs(
    location = voc_test_location,
    img_type = voc_img_type,
    num_imgs = voc_num_test_imgs, 
    width = width, 
    height = height, 
    num_channels = num_channels)

# imgs = np.load(".Object-Local-Deep-Learning/VOC_data/voc07_train.npy")
# training_imgs = imgs['arr_1']
# testing_imgs = imgs['arr_0']
#########################################################################################################
trainlabels, testlabels = get_labels()
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
n_hidden2 = 200
n_outputs = 10
learning_rate = 0.01 # grad descent param
n_epochs = 40
batch_size = 50
scale = 0.0004 # l1 regularization param
dropout_rate = 0.5
mnist = tf.contrib.learn.datasets.load_dataset("mnist")


X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
X_rs = tf.reshape(X, [-1, img_size, img_size, num_channels])
y = tf.placeholder(tf.int64, shape=(None), name="y")

dl = partial(tf.layers.dense, activation = tf.nn.relu) # dense layer

# set up graph
with tf.name_scope("dnn"):
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
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                                                            logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

# using moment descent with nesterov
with tf.name_scope("train"):
    optimizer = tf.train.MomentumOptimizer(learning_rate,momentum=0.9, use_nesterov=True)
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()
with tf.Session() as sess:
    init.run()
    save_path = saver.save(sess, "./VOC_data")
    print("Model saved in path")
    # Restore variables from disk.
    # saver.restore(sess, "/tmp/model.ckpt")
    # print("Model restored.")


    for epoch in range(n_epochs):
        start = 0
        for iteration in range(num_examples // batch_size):
            X_batch = training_imgs[start:iteration*batch_size]
            y_batch = trainlabels[start:iteration*batch_size]
            start = (start + 1)*batch_size
            sess.run(training_op,
                    feed_dict={X: X_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: testing_img,
                                            y: testlabels})
        print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)
    save_path = saver.save(sess, "./my_model_final.ckpt")



