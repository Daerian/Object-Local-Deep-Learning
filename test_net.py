import tensorflow as tf
import numpy as np
import time
from datetime import timedelta
from tensorflow.examples.tutorials.mnist import input_data
from functools import partial
import os
from PIL import Image
from PIL import ImageFilter
import scipy.misc as sm
from queue import *

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
fc_size = 2048


train = np.load("./VOC_data/voc07_train_only_blurred_nn_cropped.npy")
cv = np.load("./VOC_data/voc07_cv_only_blurred_nn_cropped.npy")

train_labels = np.load("./VOC_data/voc07_train_only_labels.npy")
cv_labels = np.load("./VOC_data/voc07_cv_only_labels.npy")

img_size = 227

num_channels = 3

# Number of classes, one class for each of 10 digits.
num_classes = 20
scale = 0.0005

training = tf.placeholder_with_default(False, shape=(), name='training')
# Batch normalization layer
bnl = partial(tf.layers.batch_normalization,
        training=training, momentum=0.99, center=True, scale=True)
# He initialization for weights to help avoid vanishing/exploding
he_init = tf.contrib.layers.variance_scaling_initializer(factor=1, mode='FAN_AVG', uniform=False)
# Dense layer
dl = partial(tf.layers.dense, activation = tf.nn.relu, kernel_regularizer=tf.contrib.layers.l1_regularizer(scale),
            kernel_initializer=he_init, use_bias=True, name=None)


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
                   use_local_norm=False, # should we use local resp. norm. for convolution layer
                   dropout_rate=0.0): # Use dropout if dropout_rate > 0

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

    if dropout_rate != 0.0:
        layer = tf.layers.dropout(layer, dropout_rate, training=training)

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

batch_size = 100
total_train_batches = 25
batch_size_cv = 100
total_cv_batchs = 25

x = tf.placeholder(tf.float32, shape=[None, img_size,img_size,num_channels], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
y0 = tf.placeholder(tf.int64, shape=(None), name='y0')

x_cv = tf.placeholder(tf.float32, shape=[None, img_size,img_size,num_channels], name='x')
x_cv_im = tf.reshape(x_cv, [-1, img_size, img_size, num_channels])
y_cv = tf.placeholder(tf.int64, shape=(None), name='y_cv')

def l_relu(z, name=None):
    return tf.maximum(0.01 * z, z, name=name)



def forw_logs (session, pre_proc_im, m5, choose):
	graph = tf.get_default_graph()

	prospects = session.run(m5, feed_dict={x: [pre_proc_im]})
	c1 = tf.slice(prospects, [0,0,0,0], [1,27,28,256])
	c1p = tf.image.resize_images(c1,[28,28])
	c2 = tf.slice(prospects, [0,0,0,0], [1,27,28,256])
	c2p = tf.image.resize_images(c1,[28,28])
	c3 = tf.slice(prospects, [0,0,0,0], [1,27,28,256])
	c3p = tf.image.resize_images(c1,[28,28])
	c4 = tf.slice(prospects, [0,0,0,0], [1,27,28,256])
	c4p = tf.image.resize_images(c1,[28,28])

	colec = np.array([c1p,c2p,c3p,c4p])
	i= 0
	logs = np.zeros(shape = [4, 20])

	for c in colec:

		flat, num_feats = flatten_layer(tf.convert_to_tensor(c1p))
		print("FLATTEN?")
		fc1_W = graph.get_tensor_by_name("fc1/kernel:0")
		fc1_b = graph.get_tensor_by_name("fc1/bias:0")
		fc1 = tf.add(tf.matmul(flat, fc1_W), fc1_b)

		fc2_W = graph.get_tensor_by_name("fc2/kernel:0")
		fc2_b = graph.get_tensor_by_name("fc2/bias:0")
		fc2 = tf.add(tf.matmul(fc1, fc2_W), fc2_b)

		outputs = graph.get_tensor_by_name("outputs/kernel:0")
		b = graph.get_tensor_by_name("outputs/bias:0")
		logits = tf.add(tf.matmul(fc2, outputs), b)
		logs[i, :] = session.run(logits)
		i += 1


	score = session.run(tf.nn.softmax(logs))
	top = score[:, CLASS]

	print ("Softmaxed Porbabilities: ")
	print (top)
	mx = np.max(top)
	print ("\nmax prob: " + str(mx) + "\n")


	cut = []

	selector = 0

	while selector < choose:

		selector += 1
				
		which = np.argmax()

		if which == 0:
			print("c1 pushed")
			cut.append(c1)
				    
		if which == 1:
			print("c2 pushed")
			cut.append(c2)
		if which == 2:
			print("c3 pushed")
			cut.append(c3)

		if which == 3:
			print("c4 pushed")
			cut.append(c4)

		top[which] = 0


	cut = np.array(cut)
	return cut


def localize(session, cls, pre_proc_im, itters, beam_width, logits, m5, f,h1,h2, split):
    
	max_loc_itters = itters
	# he_init = tf.contrib.layers.variance_scaling_initializer(factor=1, mode='FAN_AVG', uniform=False)
	# dl = partial(tf.layers.dense, activation = tf.nn.relu, kernel_regularizer=tf.contrib.layers.l1_regularizer(scale),
	#         kernel_initializer=he_init, use_bias=True, name=None)

	#logits = forw_logs(session, pre_proc_im, m5, 1)


	print("Running Localization ...")

	# print(session.run(logits, feed_dict={x: pre_proc_im}))
	# print(forw_logs (session, pre_proc_im, m5))

	cands = Queue()
	cands.put(pre_proc_im)
	CLASS = cls

	i = 0 #itteration number

	while cands.empty() == False and i < max_loc_itters:
		i += 1
		k = 0 # current beam number
		while k < beam_width:
			k+=1
			print("\n\nAttempt: " + str(i) + ", For Beam: " + str(k)) 
			if i == 1 :
				k = beam_width

			candidate = np.asarray(cands.get())
			print("Beam has found object of shape:")
			print(candidate.shape)



			#cl = forw_logs (session, candidate, m5)
			#score = session.run(tf.nn.softmax(cl))
			#top = score[:, CLASS]

			print ("Softmaxed Porbabilities: ")
			print (top)

			mx = np.max(top)
			print ("\nmax prob: " + str(mx) + "\n")


			choose = 1
			if i == 1:
				choose = beam_width

			candidate = tf.image.resize_images(candidate,[28,28])
			cut_col = forw_logs (session, candidate, m5, choose)

		    for cut in cut_col:
	
				cands.put(cut)

				if i == max_loc_itters:
					print("\n\nSaving..")
					sm.imsave("./localized_pic" + str(k) + ".jpg", cut)

def run_net(y_labs, y_true, restore):
    

    # Apply dropout to input layer
    # X_drop = tf.layers.dropout(X, 0.15, training=training)

    layer_conv1, weights_conv1 = \
        new_conv_layer(input=x_image,
                    num_input_channels=num_channels,
                    filter_size=filter_size1,
                    num_filters=num_filters1,
                    use_pooling=True,
                    use_relu=True,
                    use_local_norm=True)

    layer_conv2, weights_conv2 = \
        new_conv_layer(input=layer_conv1,
                    num_input_channels=num_filters1,
                    filter_size=filter_size2,
                    num_filters=num_filters2,
                    use_pooling=True,
                    use_relu=True,
                    use_local_norm=True)

    layer_conv3, weights_conv3 = \
            new_conv_layer(input=layer_conv2,
                num_input_channels=num_filters2,
                filter_size=filter_size3,
                num_filters=num_filters3,
                use_pooling=False,
                use_relu=True,
                use_local_norm=False,
                dropout_rate=0.0)
    bn3 = bnl(layer_conv3)

    layer_conv4, weights_conv4 = \
            new_conv_layer(input=bn3,
                num_input_channels=num_filters3,
                filter_size=filter_size4,
                num_filters=num_filters4,
                use_pooling=False,
                use_relu=True,
                use_local_norm=False,
                dropout_rate=0.0)
    bn4 = bnl(layer_conv4)

    layer_conv5, weights_conv5 = \
            new_conv_layer(input=bn4,
                num_input_channels=num_filters4,
                filter_size=filter_size5,
                num_filters=num_filters5,
                use_pooling=True,
                use_relu=True,
                use_local_norm=False)

    layer_flat, num_features = flatten_layer(layer_conv5)

    layer_fc1 = dl(layer_flat, fc_size, activation=None, name="fc1")
    bn6 = bnl(layer_fc1)
    bn6_act = tf.nn.relu(bn6)
    fc1_dropped = tf.layers.dropout(bn6_act, 0.5, training=training)

    layer_fc2 = dl(fc1_dropped, fc_size, activation=None, name="fc2")
    bn7 = bnl(layer_fc2)
    bn7_act = tf.nn.relu(bn7)
    fc2_dropped = tf.layers.dropout(bn7_act, 0.5, training=training)

    logits = dl(fc2_dropped, num_classes, activation=None, name="outputs")
    # logits = bnl(pre_logits)
    outputs = tf.nn.sigmoid(logits, name="final_activation")

    # Cross entropy cost function
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                            labels=tf.cast(tf.reshape(y_true, [batch_size, num_classes]), tf.float32))
    # get loss including regularization
    base_loss = tf.reduce_mean(cross_entropy)
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss = tf.add_n([base_loss] + reg_losses, name="loss")

    # We will be using momemtum descent with nesterov optimizationaa
    # optimizer = tf.train.MomentumOptimizer(learning_rate=0.01,momentum=0.9, use_nesterov=True)
    # Switching to Adam optimization
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001, name="optimizer")

    # Operations needed to run every iteration
    # Needed to deal with batch normalization operations
    extra_update = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    training_op = optimizer.minimize(loss)
    # Calculate accuracy; since logits is a float, it will round to 0 or 1 and then cast to int
    # for comparisons
    correct_prediction = tf.equal(tf.cast(tf.round(outputs), tf.int64), y_true)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    session = tf.Session()
    # Object to save our model after training
    saver = tf.train.Saver()
    # print(restore)
    restore = int(restore[1])
    if restore == 1:
        # path = "./datasets/VOC_2007/traindata/VOC2007/JPEGImages/"
        # gaussian_blur = 1
        # nn_size = 227

        # image_loc = path + "/000036.jpg"
        # # Open image
        # image = Image.open(image_loc)

        # image_size = image.size
        # resize_num = 500
        # if (image_size[0] > image_size[1]):
        #     resize_num = image_size[1]
        # else:
        #     resize_num = image_size[0]

        saver.restore(session, "./100_voc07_model_3.ckpt")

        # # Crop image
        # t = tf.image.resize_image_with_crop_or_pad(image, resize_num, resize_num)
        # cropped = session.run(t)

        # # Convert from numpy to PIL image
        # pil_cropped_image = Image.fromarray(cropped)

        # # Blur image
        # blurred = pil_cropped_image.filter(ImageFilter.GaussianBlur(gaussian_blur))

        # # Apply nearest neighbour interpolation
        # p = tf.convert_to_tensor(blurred)
        # nn_im = tf.image.resize_nearest_neighbor([p],
        #     (nn_size, nn_size))
        # print("Finished pre-processing image")

        # pre_proc_im = session.run(nn_im)
        # sm.imsave("./pre_proc_im.jpg", pre_proc_im[0])

        # cv_im = 101
        # local_cand = 0
        # for cv_im in range(2510):
        #     print(cv_im)
        #     pre_proc_im = cv[cv_im,:]
        #     label_true = cv_labels[cv_im,:]
        #     sm.imsave("./pre_proc_im.jpg", pre_proc_im)
            
        #     computed_logits = session.run(logits, feed_dict={x: [pre_proc_im]})
        #     # print(computed_logits)
        #     # y_class = tf.nn.sigmoid(computed_logits)
        #     y_class = session.run(tf.nn.softmax(computed_logits))
        #     print(y_class)
        #     max_cls = np.argmax(y_class)
        #     print(max_cls)
        #     y_class = np.zeros(shape=(1, 20))
        #     y_class[0,max_cls] = 1
        #     # y_pred = session.run(tf.cast(tf.round(y_class), tf.int64))
        #     # y_pred_probs = session.run(y_class)
        #     print("True labels:")
        #     print(label_true)
        #     print("Prediction:")
        #     print(y_class)
        #     # print(y_pred_probs)
        #     # print()
        #     if np.sum((y_class == 1) == (label_true == 1)) == 20:
        #         print(cv_im)
        #         local_cand = cv_im
        #         break

        
        # LOCALIZATION

        #split into 4
        # pre_proc_im = np.zeros(shape=(1,227,227,3))
        # pre_proc_im[0,:,:,:] = cv[95,:]
        pre_proc_im = cv[95,:]

        cands = Queue()
        cands.put(pre_proc_im)
        CLASS = 14
        localize(session, CLASS, pre_proc_im, 80, 2, logits, 
                layer_conv5, layer_flat, layer_fc1, layer_fc2, 1)

        
    else:
        init = tf.global_variables_initializer()
        session.run(init)

        # Ensure we update the global variable rather than a local copy.
        total_iterations = 301

        # Put the datasets into Tensorflow's Dataset object
        train_dataset = tf.data.Dataset.from_tensor_slices((x,y_true)).repeat().batch(batch_size)
        cv_dataset = tf.data.Dataset.from_tensor_slices((x_cv_im,y_cv)).repeat().batch(batch_size_cv)
        # Iterator for Datasets
        it = train_dataset.make_initializable_iterator()
        it_cv = cv_dataset.make_initializable_iterator()

        session.run(it.initializer, feed_dict={x:train, y_true: y_labs})
        session.run(it_cv.initializer, feed_dict={x_cv:cv, y_cv: cv_labels})

        start_time = time.time()
        # Iteratior object to get every batch in for loop
        x_batch, y_true_batch = it.get_next()
        x_cv_batch, y_cv_batch = it_cv.get_next()

        for i in range(total_iterations):
            print("iteration: " + str(i))

            X_eval, y_eval = session.run([x_batch, y_true_batch])

            feed_dict_train = {training: True, x: X_eval,
                            y_true: y_eval}

            session.run([training_op, extra_update], feed_dict=feed_dict_train)

            print("Labels:")
            print(y_eval[0:20,:])
            print("Logits:")
            print(session.run(logits, feed_dict={x: X_eval[0:20,:], y_true: y_eval[0:20,:]}))
            # Print status every 5 iterations
            if i % 5 == 0:
                # Calculate the accuracy on the training-set
                acc = session.run(accuracy, feed_dict=feed_dict_train)
                msg = "Optimization Iteration: " +str(i+1)+", Training Accuracy: " + str(acc)
                print(msg)

            if i % total_train_batches == 0 and i != 0:
                print("Checkpoint..")
                save_path = saver.save(session, "./" + str(i) +  "_voc07_model_3.ckpt")
                total = 0
                for j in range(total_cv_batchs):
                    print(str(j) + ": Computing CV Accuracy..")
                    x_eval_cv, y_cv_eval = session.run([x_cv_batch, y_cv_batch])
                    cv_acc = session.run(accuracy, feed_dict={x: x_eval_cv,
                                                            y_true: y_cv_eval})
                    print("Cross Validation Accuracy: " + str(cv_acc))
                    total += cv_acc
                print("CV Average: " + str(total/float(total_cv_batchs)))

        # Ending time
        end_time = time.time()
        save_path = saver.save(session, "./voc07_model_3.ckpt")
        # Difference between start and end-times
        time_dif = end_time - start_time

        # Print the time-usage.
        print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
    session.close()


def main(if_restore=0):
    c0 = train_labels
    w0 = run_net(c0, y0, restore=if_restore)

if __name__ == '__main__':
    tf.app.run(main=main)
