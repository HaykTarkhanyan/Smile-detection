import cv2
from model import load_model
from helpers import process_input
from random import randint

THRESHOLD = 0.5
MODEL_DIR = ""

# https://github.com/GangYuanFan/Closed-Eye-Detection-with-opencv/blob/master/cv_close_eye_detect.py
eye_cascPath = r"C:/Users/Hayk/Desktop/haarcascade_eye_tree_eyeglasses.xml"
face_cascPath = r"C:/Users/Hayk/Desktop/haarcascade_frontalface_default.xml"

faceCascade = cv2.CascadeClassifier(face_cascPath)
eyeCascade = cv2.CascadeClassifier(eye_cascPath)

# added just for fun
emoj = cv2.imread("emoji.png")
emoji = cv2.resize(emoj, (150, 150))


loaded_model = load_model()

cap = cv2.VideoCapture(0)

done = False

while not done:
    ret, img = cap.read()

    if ret:
        frame_to_display = img.copy()
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
            # for (x, y, w, h) in faces:
            #     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

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
                print("Open your eyes. I don't know how are you gonna read this with closed eyes, though")
            else:
                pred = loaded_model.predict(
                    process_input('ban', frame_cropped))
                print(pred)
                if pred > THRESHOLD:
                    cv2.imwrite(f'smile_{randint(1,10)}.png', frame_to_display)
                    frame_to_display[:emoj.shape[0], :emoj.shape[1]] = emoj


            cv2.imwrite("cropped_face.png", frame_cropped)
            cv2.imshow('Face Recognition', frame_to_display)

        waitkey = cv2.waitKey(1)
        if waitkey == ord('q') or waitkey == ord('Q'):
            cv2.destroyAllWindows()
            done = True
            break
