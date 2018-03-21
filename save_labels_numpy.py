import tensorflow as tf
import numpy as np

def main(unused_arg):
    filename_queue = tf.train.string_input_producer(["./VOC_data/train_only_labels.csv", "./VOC_data/test_labels.csv"])
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

        # trainlabels = np.array([])
        # testlabels = np.array([])
        trainlabels = np.zeros(shape=(2501, 20))
        # testlabels = np.zeros(shape=(4952, 20))
        for i in range(2501):
            # Retrieve a single instance:
            print("train obs: " + str(i))
            set1 = sess.run(multi_trainlabels)
            # trainlabels = np.append(trainlabels, example)
            trainlabels[i] = set1
            trainlabels[i] = ((set1 +1) /2)
            trainlabels[i][trainlabels[i] == 0.5] = 0

            # print(example)
            # print(type(multi_labels))
        
        # for i in range(4952):
        #     # Retrieve a single instance:
        #     print("test obs: " + str(i))
        #     set1 = sess.run(multi_testlabels)
        #     # np.append(testlabels, example)
        #     testlabels[i] = set1
        #     testlabels[i] = ((set1 +1) /2)
        #     testlabels[i][testlabels[i] == 0.5] = 0

    coord.request_stop()
    coord.join(threads)
    print("saving..")
    np.save('./VOC_data/voc07_train_only_labels.npy', trainlabels)
    # np.save('./VOC_data/voc07_test_labels3.npy', testlabels)
    print("saved")


if __name__ == "__main__":
    tf.app.run(main=main)
