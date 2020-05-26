import cv2
import numpy as np
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator

EPOCHS = 3


def process_input(direc, photo=None):
    if photo is None:
        test_im = cv2.imread(direc)
    else:
        test_im = photo
    gray = cv2.cvtColor(test_im, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (64, 64))
    gray = np.expand_dims(gray, axis=2)
    gray = np.expand_dims(gray, axis=0)
    return gray


# plt.imshow(one_0.flatten().reshape(64, 64), cmap='gray')
# plt.imshow(zero_0.flatten().reshape(64, 64), cmap='gray')
# plt.imshow(zero_1.flatten().reshape(64, 64), cmap='gray')

# plt.show()


# model = Sequential()
# # Layer 1
# # Conv Layer 1
# model.add(Conv2D(filters=6,
#                  kernel_size=5,
#                  strides=1,
#                  activation='relu',
#                  input_shape=(64, 64, 1)))
# # Pooling layer 1
# model.add(MaxPooling2D(pool_size=2, strides=2))
# # Layer 2
# # Conv Layer 2
# model.add(Conv2D(filters=16,
#                  kernel_size=5,
#                  strides=1,
#                  activation='relu'))
# #  input_shape = (14,14,6)))
# # Pooling Layer 2
# model.add(MaxPooling2D(pool_size=2, strides=2))
# # Flatten
# model.add(Flatten())
# # Layer 3
# # Fully connected layer 1
# model.add(Dense(units=120, activation='relu'))
# # Layer 4
# # Fully connected layer 2
# model.add(Dense(units=84, activation='relu'))
# # Layer 5
# # Output Layer
# model.add(Dense(units=1, activation='sigmoid'))

# # print (model.summary())
# model.compile(optimizer='adam', loss='binary_crossentropy',
#               metrics=['accuracy'])

# history = model.fit_generator(
#     train_generator,
#     steps_per_epoch=train_generator.samples // batch_size,
#     validation_data=validation_generator,
#     validation_steps=validation_generator.samples // batch_size,
#     epochs=EPOCHS)
# # model.fit(X_train ,Y_train, steps_per_epoch = 10, epochs = 42)

# print(model.predict(one_0))
# print(model.predict(zero_0))
# print(model.predict(zero_1))
# # , zero_0, zero_1])
# # print (y_pred)
