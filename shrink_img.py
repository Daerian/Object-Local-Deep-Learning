import tensorflow as tf
import numpy as np
import os
import matplotlib.image as mpimg

def main(unused_arg):


    img_size = 150


    
    # print ("loading training imgs")
    # traindata = np.load("./VOC_data/voc07_train1.npy")
    # sess = tf.InteractiveSession()

    # l = traindata.shape

    # train_cropped = np.zeros(shape=(5011,img_size,img_size,3))
    # print("begin")
    # for i in range(l[0]):
    #     print(i)
    #     p = tf.convert_to_tensor(traindata[i], np.float32)
    #     t = tf.image.resize_nearest_neighbor([p],
    #         (img_size, img_size))
    #     print(t[0].shape)
    #     train_cropped[i] = t[0].eval()

    # sess.close()
    # np.save("./VOC_data/voc07_train_cropped.npy", train_cropped)


    print ("loading testing imgs")
    testdata = np.load("./VOC_data/voc07_test.npy")
    sess = tf.InteractiveSession()

    l = testdata.shape
    test_cropped = np.zeros(shape=(4952,img_size,img_size,3))
    print("begin")
    for i in range(l[0]):
        print(i)
        p = tf.convert_to_tensor(testdata[i], np.float32)
        t = tf.image.resize_nearest_neighbor([p],
            (img_size, img_size))
        # print(t[0].shape)
        test_cropped[i] = t[0].eval()
    np.save('./VOC_data/voc07_test_cropped.npy', test_cropped)
    sess.close()



if __name__ == "__main__":
    tf.app.run(main=main)

