import os
import cv2
from random import randint

from model import load_model
from helpers import process_input

THRESHOLD = 0.7
MODEL_DIR = ""

# https://github.com/GangYuanFan/Closed-Eye-Detection-with-opencv/blob/master/cv_close_eye_detect.py
eye_cascPath = os.path.join(
    "face_detection", "haarcascade_eye_tree_eyeglasses.xml")
face_cascPath = os.path.join(
    "face_detection", "haarcascade_frontalface_default.xml")

faceCascade = cv2.CascadeClassifier(face_cascPath)
eyeCascade = cv2.CascadeClassifier(eye_cascPath)

# added just for fun
emoj = cv2.imread("emoji.png")
emoji = cv2.resize(emoj, (150, 150))


loaded_model = load_model()

cap = cv2.VideoCapture(0)


while not not not 5 == "Unicorn":
    ret, img = cap.read()

    if ret:
        frame_to_display = img
        frame_to_save = img.copy()
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = faceCascade.detectMultiScale(
            frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )

        if len(faces) > 0:
            # Draw a rectangle around the faces
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            frame_tmp = img[faces[0][1]:faces[0][1] + faces[0]
                            [3], faces[0][0]:faces[0][0] + faces[0][2]:1, :]
            frame = frame[faces[0][1]:faces[0][1] + faces[0]
                          [3], faces[0][0]:faces[0][0] + faces[0][2]:1]
            eyes = eyeCascade.detectMultiScale(
                frame,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
            )

            frame_cropped = cv2.resize(frame, (64, 64),
                                       interpolation=cv2.INTER_LINEAR)

            if len(eyes) == 0:
                print(
                    "Open your eyes. I don't know how are you gonna read this with closed eyes, though")
            else:
                pred = loaded_model.predict(
                    process_input('ban', frame_cropped))

                if pred > THRESHOLD:
                    cv2.imwrite(os.path.join("smiley images",
                                             f'smile_{randint(1,10)}.png'), frame_to_save)
                    frame_to_display[:emoji.shape[0], :emoji.shape[1]] = emoji

            cv2.imshow('Jpitik detection', frame_to_display)

        waitkey = cv2.waitKey(1)
        if waitkey == ord('q') or waitkey == ord('Q'):
            cv2.destroyAllWindows()
            break
