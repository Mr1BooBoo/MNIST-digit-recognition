# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 03:33:20 2021

@author: Orochi
"""
import os
import cv2 as cv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#get the data
mnist = tf.keras.datasets.mnist


#divide the data in train and test sets
(x_train, y_train),(x_test, y_test) = mnist.load_data()



#normalize the images x train/test (scale it down from gray scale 0-255 to 0-1)
x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)

#initialize neural model
model = tf.keras.models.Sequential()

#flat the images from 28x28 to 784
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
#dense layer all connects to all
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))

#selecting optimizer, loss function
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


model.fit(x_train, y_train, epochs= 10)

for layer in model.layers:
    print(layer.output_shape)


accuracy, loss= model.evaluate(x_test, y_test)

model.save('digits.model')
print(accuracy)
print(loss)




#################### predict hand written images from a directory on pc ####################

#getting the directory and nr of images 
list = os.listdir('C:/Users/bilal/OneDrive/Desktop/Code/MNIST_numbers') # dir is your directory path
number_imgs = len(list)
print(number_imgs)


path = 'C:/Users/bilal/OneDrive/Desktop/Code/MNIST_numbers/'

#parse images and feed them to the NN
for x in range(number_imgs):
    img_name = f'img{x}.png'
    final_path = "{}{}".format(path, img_name)
    img = cv.imread(final_path, -1)[:,:,0]
    img = np.invert(np.array([img]))
    prediction = model.predict(img)
    print(f'you typed a {np.argmax(prediction)}')
    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.show()















