import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.models import model_from_json
import cv2
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from keras.regularizers import l2
