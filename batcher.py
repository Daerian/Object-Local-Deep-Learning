import tensorflow as tf
import numpy as np




def main(unused_arg):

    # LOAD TRAINING AND TESTING DATA
    train_data = np.load("./VOC_data/voc07_train_padded.npy")
    #test_data = np.load("./VOC_data/voc07_test_padded.npy")
    #train_labels = np.load("./VOC_data/voc07_train_labels.npy")
    #test_labels = np.load("./VOC_data/voc07_test_labels.npy")

    # # CREATE INTERACTIVE SESSION
    # sess = tf.InteractiveSession()

    # # CREATE DATASETS FOR BOTH TRAINING AND TESTING
    # #train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
    # #test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_labels))
    
    # place = tf.placeholder(tf.float32, shape=(5011, 500, 500, 3))
    # Train_DB = tf.Variable(place)
    # sess.run(tf.global_variables_initializer(), feed_dict={place: train_data})   
    # # END

    # print ("make dataset")
    # train_dataset = tf.data.Dataset.from_tensor_slices((Train_DB))
    # # train_dataset = tf.data.Dataset.zip((train_data))

    # print("try to batch")
    # z = train_dataset.batch(100)
    
    # print(z)
    # itert = train_dataset.make_one_shot_iterator()
    # next_stuff = itert.get_next()
    # print(next_stuff.eval())

    # # Y = tf.train.batch(
    # #     tensors = train_dataset.batch(100),
    # #     batch_size = 100)
    # sess.close()

    train1_data = train_data[0:1000]
    # train2_data = train_data[2511:5011]


    sess = tf.InteractiveSession()
    init = tf.global_variables_initializer()
    init.run()

    train1_dataset = tf.data.Dataset.from_tensor_slices((train1_data))
    z = train1_dataset.batch(100)
    print(z)

    itert = train1_dataset.make_one_shot_iterator()
    stuff = itert.get_next().eval()
    print(stuff[0][225][100])

    sess.close()



if __name__ == "__main__":
    tf.app.run(main=main)