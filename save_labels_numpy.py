import tensorflow as tf
import numpy as np

def main(unused_arg):
    filename_queue = tf.train.string_input_producer(["./VOC_data/cv_only_labels.csv"])#, "./VOC_data/cv_only_labels.csv"])
    reader = tf.TextLineReader()
    # train_key, train_vals = reader.read(filename_queue)
    cv_key, cv_vals = reader.read(filename_queue)

    # Default values, in case of empty columns. Also specifies the type of the
    # decoded result.
    record_defaults = [[1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1]]
    # name,aeroplane,bicycle,bird,boat,bottle,bus,car,cat,chair,cow,diningtable,dog,horse,motorbike,person,pottedplant,sheep,sofa,train,tvmonitor = tf.decode_csv(train_vals, record_defaults=record_defaults)
    # multi_trainlabels = tf.stack([aeroplane,bicycle,bird,boat,bottle,bus,car,cat,chair,cow,diningtable,dog,horse,motorbike,person,pottedplant,sheep,sofa,train,tvmonitor])

    cv_name,cv_aeroplane,cv_bicycle,cv_bird,cv_boat,cv_bottle,cv_bus,cv_car,cv_cat,cv_chair,cv_cow,cv_diningtable,cv_dog,cv_horse,cv_motorbike,cv_person,cv_pottedplant,cv_sheep,cv_sofa,cv_train,cv_tvmonitor = tf.decode_csv(cv_vals, record_defaults=record_defaults)
    multi_cvlabels = tf.stack([cv_aeroplane,cv_bicycle,cv_bird,cv_boat,cv_bottle,cv_bus,cv_car,cv_cat,cv_chair,cv_cow,cv_diningtable,cv_dog,cv_horse,cv_motorbike,cv_person,cv_pottedplant,cv_sheep,cv_sofa,cv_train,cv_tvmonitor])

    train_amount = 2501
    cv_amount = 2510

    with tf.Session() as sess:
    # Start populating the filename queue.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # trainlabels = np.zeros(shape=(train_amount, 20))
        cvlabels = np.zeros(shape=(cv_amount, 20))
        # for i in range(train_amount):
        #     # Retrieve a single instance:
        #     print("train obs: " + str(i))
        #     set1 = sess.run(multi_trainlabels)
        #     trainlabels[i] = set1
        #     trainlabels[i] = ((set1 +1) /2)
        #     trainlabels[i][trainlabels[i] == 0.5] = 0

        for i in range(cv_amount):
            # Retrieve a single instance:
            print("cv obs: " + str(i))
            set1 = sess.run(multi_cvlabels)
            cvlabels[i] = set1
            cvlabels[i] = ((set1 +1) /2)
            cvlabels[i][cvlabels[i] == 0.5] = 0

    coord.request_stop()
    coord.join(threads)
    print("saving..")
    # np.save('./VOC_data/voc07_train_only_labels.npy', trainlabels)
    np.save('./VOC_data/voc07_cv_only_labels.npy', cvlabels)
    print("saved")


if __name__ == "__main__":
    tf.app.run(main=main)
