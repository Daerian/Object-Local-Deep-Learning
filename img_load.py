# from PIL import Image
# import glob
import numpy as np
import tensorflow as tf


# a function to load images froma folder into a tensorflow dataset
def load_imgs(location, img_type, num_imgs, width, height,num_channels):
  location = location + "*." + img_type
  print("reading from: " + location)
  # location = location + "000005.jpg"
  #The queue for all the files in the firectory
  voc_queue = tf.train.string_input_producer(
  tf.train.match_filenames_once(location))
  # print(voc_queue.dequeue_many(5012))


  #IMG reader, stores in string format
  image_reader = tf.WholeFileReader()

  #image tensor for read images above
  img_tensor = []#img_tensor = []

  print("decoding images ...")
  
  # begin decoding
  for i in range(num_imgs):
    print(i)
    #Reads the file at the top of the queue, first arg is name, irrellevant
    _, image_file = image_reader.read(voc_queue)
    print(image_file)
    #Decoder that turns the above img reader string into a tensor
    voc_image = tf.image.decode_jpeg(
      contents = image_file,
      channels = num_channels)

    img = tf.image.resize_image_with_crop_or_pad(
      image = voc_image, # current image
      target_height = height, #given width
      target_width = width #given height
    )
    print(img)

    #add to list of tensors
    img_tensor.append(img)
  

  data = np.array(img_tensor)
  print(data)
  return (data)


def main(unused_arg):
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
    num_channels = 3)

  np.save('./VOC_data/voc07_train.npy',training_imgs)

  voc_num_test_imgs = 4952
  voc_test_location = "./datasets/VOC_2007/testdata/VOC2007/JPEGImages/"

  # call the img_loader
  testing_img = load_imgs(
      location = voc_test_location,
      img_type = voc_img_type,
      num_imgs = voc_num_test_imgs, 
      width = width, 
      height = height, 
      num_channels = 3)

  np.save('./VOC_data/voc07_test.npy', testing_img)


if __name__ == "__main__":
  tf.app.run(main=main)
