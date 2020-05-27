import cv2
import numpy as np

def process_input(dir = None, photo=None, from_webcam=True):
    """
    Function covnerts image to format acceptable by the model

    Note: If image comes from webcam part of preprocessing is already done

    Args:
    dir - directory of the file(str)
    photo - image itself, already loaded(ndarray)
    from_webacm - bool
    """
    if photo is None:
        test_im = cv2.imread(dir)
    else:
        test_im = photo
    if from_webcam:
        gray = test_im
    else:
        gray = cv2.cvtColor(test_im, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (64, 64))
    gray = test_im
    gray = np.expand_dims(gray, axis=2)
    gray = np.expand_dims(gray, axis=0)
    return gray

