import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import cv2 
import imutils
import matplotlib.pyplot as plt
from os import listdir
import time 
import numpy as np

def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return f"{h}:{m}:{round(s, 1)}"

def augment_data(file_dir, n_generated_samples, save_to_dir):

    data_gen = ImageDataGenerator(rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    brightness_range=(0.3, 1.0),
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
    )

    for filename in listdir(file_dir):
        image  = cv2.imread(file_dir + '/' + filename)
        # image = image.resize((32, 32))
        # image = np.array(image)
        image = image.reshape((1,)+image.shape)

        save_prefix = 'aug_' + filename[:-4]

        i = 0

        for batch in data_gen.flow(x=image, batch_size=1, save_to_dir = save_to_dir, save_prefix=save_prefix, save_format='jpg'):
            i += 1
            if i > n_generated_samples:
                break


start_time = time.time()

augmented_data_path = '/home/nikhil/brain_tumor/augmented data/'

augment_data(file_dir='/home/nikhil/brain_tumor/yes', n_generated_samples=6, save_to_dir=augmented_data_path+'yes')

augment_data(file_dir='/home/nikhil/brain_tumor/no', n_generated_samples=9, save_to_dir=augmented_data_path+'no')

end_time = time.time()
execution_time = end_time - start_time

print(f"Elapsed time: {hms_string(execution_time)}")

def data_summary(main_path):

    yes_path = main_path+'yes'
    no_path = main_path+'no'

    m_pos = len(listdir(yes_path))

    m_neg = len(listdir(no_path))

    m = m_pos + m_neg

    pos_precentage = (m_pos/m)*100
    neg_precentage = (m_neg/m)*100

    print(f"Total number of images: {m}")
    print(f"Positive images: {m_pos}, Percentage: {pos_precentage}%")
    print(f"Negative images: {m_neg}, Percentage: {neg_precentage}%")

data_summary(augmented_data_path)