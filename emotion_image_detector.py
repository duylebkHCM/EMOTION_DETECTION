from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import cv2
import os
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-t", "--testImage", required=True, help="path to test Images")
ap.add_argument("-m", "--model", required=True, help="path to model checkpoint")
args = vars(ap.parse_args())

path = str(args['testImage'])
listImage = os.listdir(args['testImage'])

model = load_model(args['model'])
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
EMOTIONS = ["angry", "scary", "happy", "sad", "supprise", "neutral"]

for imageName in listImage:
    image = cv2.imread(path + '/' + imageName)
    image = cv2.resize(image, (300, 300))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50), flags=cv2.CASCADE_SCALE_IMAGE)
    
    if len(rects) > 0:
        rect = sorted(rects, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = rect
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        with tf.device('/CPU:0'):
            preds = model.predict(roi)[0]
        label = EMOTIONS[preds.argmax()]

        cv2.putText(image, label, (fX, fY - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.rectangle(image, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)

    cv2.imshow("Emtion", image)
    cv2.waitKey(0)

cv2.destroyAllWindows()

