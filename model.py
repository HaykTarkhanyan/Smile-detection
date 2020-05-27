import os

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

from keras.models import model_from_json



def le_net_modified(LAMBD):
        model = Sequential()

        model.add(Conv2D(filters = 6,
                         kernel_size = 5,
                         strides = 1,
                         activation = 'relu',
                         input_shape = (64,64,1),
                         kernel_regularizer=l2(LAMBD),
                         bias_regularizer=l2(LAMBD)))

        model.add(MaxPooling2D(pool_size = 2, strides = 2, ))

        model.add(Conv2D(filters = 16,
                         kernel_size = 5,
                         kernel_regularizer=l2(LAMBD),
                         bias_regularizer=l2(LAMBD),
                         strides = 1,
                         activation = 'relu'))


        model.add(MaxPooling2D(pool_size = 2, strides = 2))

        model.add(Flatten())
        model.add(Dense(units = 120,
                        activation = 'relu',
                        kernel_regularizer=l2(LAMBD),
                        bias_regularizer=l2(LAMBD)))

        model.add(Dense(units = 84,
                        activation = 'relu',
                        kernel_regularizer=l2(LAMBD),
                        bias_regularizer=l2(LAMBD)))

        model.add(Dense(units = 1, activation = 'sigmoid'))

        return model

def save_model(save_dir):
    # serialize model to JSON
    model_json = model.to_json()
    with open(f"/content/drive/My Drive/Smile_dataset/smile/ep_{EPOCHS} l2_{lambd}.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(f"/content/drive/My Drive/Smile_dataset/smile/ep_{EPOCHS} l2_{lambd}.h5")
    print("Saved model to disk")

    return model

def load_model(model_loc=os.path.join("weights and config", "model_l2_0.01")):
    """
    Load h5 and json files from given dircetory

    Args:
    model_loc - folder where configuration and wieghts are stoerd
    """
    # load json and create model
    json_file = open(f"{model_loc}.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(f"{model_loc}.h5")
    print("Loaded model")

    return loaded_model
