import gzip
import numpy as np
import struct

def loadData(images_file_path, labels_file_path):
  with gzip.open(images_file_path, 'rb') as f:
    magic_number_img = struct.unpack('>I', f.read(4))[0]

    if magic_number_img != 2051:
      raise ValueError('Traing images dataset: Expected value of magic value is 2051, got {}'.format(magic_number_img))

    num_of_imgs = struct.unpack('>I', f.read(4))[0]
    x_size = struct.unpack('>I', f.read(4))[0]
    y_size = struct.unpack('>I', f.read(4))[0]
    train_imgs_data = f.read()

  imgs_data_array = np.frombuffer(train_imgs_data, dtype=np.uint8)
  imgs = []

  for i in range(num_of_imgs):
    img = np.array(imgs_data_array[i * x_size * y_size : (i + 1) * x_size * y_size])
    img = img.reshape(x_size * y_size)
    img[img != 0] = 1
    imgs.append(img)

  imgs = np.array(imgs)


  with gzip.open(labels_file_path, 'rb') as f:
    magic_number_label = struct.unpack('>I', f.read(4))[0]

    if magic_number_label != 2049:
      raise ValueError('Traing labels dataset: Expected value of magic value is 2049, got {}'.format(magic_number_label))

    num_of_labels = struct.unpack('>I', f.read(4))[0]

    if num_of_labels != num_of_imgs:
      raise ValueError('The number of images has to be equal to the number of labels.\nNumber of images = {}\nNumber of labels = {}'.format(num_of_imgs, num_of_labels))
    train_labels_data = f.read()

  labels = np.frombuffer(train_labels_data, dtype=np.uint8)

  return imgs, labels