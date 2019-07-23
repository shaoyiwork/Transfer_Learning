#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.models import load_model


model = Sequential()
#load h5 model files
model=load_model("transfer_model.h5")
model.summary()
#read the test image
img = image.load_img(path="./test/horse1.jpg", target_size=(234,234,3))
img = image.img_to_array(img)
img = img.astype('float32')
img /= 255
test_img = img.reshape((1,234,234,3))
preds = model.predict(test_img)
print (preds[0])
#get the predict class
labels =  ['cat','dog','horse']
Final_prediction = [result.argmax() for result in preds][0]
Final_prediction = labels[Final_prediction]
print("class :",Final_prediction)
if Final_prediction == 'cat':
    title = 'cat'
elif Final_prediction == 'dog':
    title = 'dog'
elif Final_prediction == 'horse':
    title = 'horse'
#show the result
plt.imshow(img)
plt.title(title)
plt.show()
