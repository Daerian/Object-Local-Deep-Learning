import tensorflow as tf
import numpy as np
import os
import matplotlib.image as mpimg

def main(unused_arg):
    
    print ("loading training imgs")
    traindata = np.load("./VOC_data/voc07_train1.npy")
    sess = tf.InteractiveSession()

    l = traindata.shape

    train_cropped = np.zeros(shape=(5011,50,50,3))
    print("begin")
    for i in range(l[0]):
        print(i)
        p = tf.convert_to_tensor(traindata[i], np.float32)
        t = tf.image.resize_nearest_neighbor([p],
            (50, 50))
        print(t.shape)
        train_cropped[i] = t.eval()
    # for i in range(l[0]):
    #    print(i)
    #    t = tf.image.resize_nearest_neighbor(
    #        image = traindata[i], 
    #        size = [50, 50],
    #        align_corners=True)

    #    padded = t.eval()
    #    print(padded.shape)
    #    train_padded[i] = padded
    sess.close()
    np.save("./VOC_data/voc07_train_cropped.npy", train_cropped)


    print ("loading testing imgs")
    testdata = np.load("./VOC_data/voc07_test.npy")
    sess = tf.InteractiveSession()

    l = testdata.shape
    #test_padded = np.zeros(shape=(5011,50,50,3))
    print("begin")
    t = tf.image.resize_nearest_neighbor(
           images = testdata, 
           size = [50, 50])
    test_padded = t.eval()
    # for i in range(l[0]):
    #     print(i)
    #     t = tf.image.resize_nearest_neighbor(
    #        image = testdata[i], 
    #        size = [50, 50],
    #        align_corners=True)

    #     padded = t.eval()
    #     print(padded.shape)
    #     test_padded = np.append(test_padded, padded)
    np.save('./VOC_data/voc07_test_padded2.npy', test_padded)
    sess.close()



if __name__ == "__main__":
    tf.app.run(main=main)

