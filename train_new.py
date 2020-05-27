import cv2
import numpy as np
import matplotlib.pyplot as plt

import keras
from keras.preprocessing.image import ImageDataGenerator

from model import le_net_modified, save_model


VALID_SPLIT = 0.1

EPOCHS = 20
BATCH_SIZE = 16

LAMBD = 0.005

DATA_DIR = "/content/drive/My Drive/Smile_dataset/smile/lfwcrop_grey/data"

train_datagen = ImageDataGenerator(rescale=1./255,
                                   validation_split=VALID_SPLIT
                                   )

train_generator = train_datagen.flow_from_directory(
                                                    directory=DATA_DIR,
                                                    target_size=(64, 64),
                                                    color_mode="grayscale",
                                                    batch_size=BATCH_SIZE,
                                                    class_mode="binary",
                                                    shuffle=True,
                                                    seed=42,
                                                    subset = "training"
                                                    )

validation_generator = train_datagen.flow_from_directory(
                                                    directory=DATA_DIR,
                                                    target_size=(64, 64),
                                                    batch_size=BATCH_SIZE,
                                                    color_mode="grayscale",
                                                    class_mode='binary',
                                                    subset='validation')
model = le_net_modified(LAMBD)

print (model.summary())

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

history = model.fit_generator(
                    train_generator,
                    steps_per_epoch = train_generator.samples // BATCH_SIZE,
                    validation_data = validation_generator,
                    validation_steps = validation_generator.samples // BATCH_SIZE,
                    epochs = EPOCHS
                    )



# y_pred = np.argmax(Y_pred, axis=1)
# print(f'Confusion Matrix{i}')
# print(confusion_matrix(validation_generator.classes, y_pred))
# print(f'Classification Report {i}')

# print(classification_report(validation_generator.classes, y_pred))


# serialize model to JSON
save_model()


