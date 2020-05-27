import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import cv2
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from keras.regularizers import l2
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
                    steps_per_epoch = train_generator.samples // batch_size,
                    validation_data = validation_generator,
                    validation_steps = validation_generator.samples // batch_size,
                    epochs = EPOCHS
                    )



# y_pred = np.argmax(Y_pred, axis=1)
# print(f'Confusion Matrix{i}')
# print(confusion_matrix(validation_generator.classes, y_pred))
# print(f'Classification Report {i}')

# print(classification_report(validation_generator.classes, y_pred))


# serialize model to JSON
model_json = model.to_json()
with open(f"/content/drive/My Drive/Smile_dataset/smile/ep_{EPOCHS} l2_{lambd}.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(f"/content/drive/My Drive/Smile_dataset/smile/ep_{EPOCHS} l2_{lambd}.h5")
print("Saved model to disk")



# # later...

# # load json and create model
# json_file = open('model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# # load weights into new model
# loaded_model.load_weights("/content/drive/My Drive/Smile_dataset/smile/model.h5")
# print("Loaded model from disk")


# , zero_0, zero_1])
# print (y_pred)
