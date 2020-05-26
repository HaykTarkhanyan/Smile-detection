import cv2
from keras.models import model_from_json
from train import process_input

# https://github.com/GangYuanFan/Closed-Eye-Detection-with-opencv/blob/master/cv_close_eye_detect.py
eye_cascPath = r"C:/Users/Hayk/Desktop/haarcascade_eye_tree_eyeglasses.xml"
face_cascPath = r"C:/Users/Hayk/Desktop/haarcascade_frontalface_default.xml"

faceCascade = cv2.CascadeClassifier(face_cascPath)
eyeCascade = cv2.CascadeClassifier(eye_cascPath)

threshold = 0.5

emoj = cv2.imread("emoji.png")
emoj = cv2.resize(emoj, (150, 150))

print(emoj.shape)

# load json and create model
json_file = open('model_l2_0,01.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model_l2_0.01.h5")
print("Loaded model from disk")

cap = cv2.VideoCapture(0)
while 1:
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

            frame_cropped = cv2.resize(frame_tmp, (64, 64),
                                       interpolation=cv2.INTER_LINEAR)

            if len(eyes) == 0:
                print('no eyes!!!')
            else:
                pred = loaded_model.predict(
                    process_input('ban', frame_cropped))
                # cv2.imwrite("")
                # if pred >= 0.5:
                # pred = 0.6
                print(pred)
                if pred > threshold:
                    frame_to_display[:emoj.shape[0], :emoj.shape[1]] = emoj

                # print('eyes!!!')

            cv2.imwrite("cropped_face.png", frame_cropped)

            cv2.imshow('Face Recognition', frame_to_display)

            # cv2.imshow('Face Recognition', frame)

        waitkey = cv2.waitKey(1)
        if waitkey == ord('q') or waitkey == ord('Q'):
            cv2.destroyAllWindows()
            break
