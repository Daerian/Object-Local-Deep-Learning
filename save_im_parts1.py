import tensorflow as tf
import numpy as np
import os
# import cv2
import matplotlib.image as mpimg


def main(unused_arg):

    traindata = np.load("./VOC_data/voc07_train1.npy")
    traindata = traindata[:-3]
    traindata = np.split(traindata, 16)
    #split into 16 sets

    sess = tf.InteractiveSession()

    l = traindata[0].shape
    print(l)
    #pad and save the first sets
    #train_padded = np.array([])
    train_padded = np.zeros(shape=(313, 500, 500, 3))
    for i in range(l[0]):
       print(i)
       #t = tf.image.resize_image_with_crop_or_pad(traindata[0][i], 500, 500)
       #padded = t.eval()
       #print(padded.shape)
       train_padded = np.append(train_padded, traindata[0][i])
    train_padded2 = tf.reshape(train_padded, [313, 500, 500, 3])
    train_padded = train_padded2.eval()
    print(train_padded.shape)
    sess.close()
    #np.save("./VOC_data/voc07_train_padded@0.npy", train_padded)


    # testdata = np.load("./VOC_data/voc07_test.npy")
    # sess = tf.InteractiveSession()

    # l = testdata.shape
    #  test_padded = np.array([])
    # for i in range(l[0]):
    #     print(i)
    #     t = tf.image.resize_image_with_crop_or_pad(testdata[i], 500, 500)
    #     padded = t.eval()
    #     print(padded.shape)
    #     test_padded = np.append(test_padded, padded)
    # np.save('./VOC_data/voc07_test_padded.npy', test_padded)
    # sess.close()



if __name__ == "__main__":
    tf.app.run(main=main)
