#!/usr/bin/env python3

import cv2
import tensorflow.keras
import numpy as np
import time

MODEL_PATH = 'models/keras_model.h5'
LABELS_PATH = 'models/labels.txt'


def read_model(model_path, labels_path):
  np.set_printoptions(suppress=True)
  model = tensorflow.keras.models.load_model(model_path)

  label_file = open(labels_path, "r")
  labels = [x.strip() for x in label_file]
  return model, labels

def predict(img, model, labels):
  data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
  img_arr = np.asarray(cv2.resize(img, (224, 224)))
  data[0]  = (img_arr.astype(np.float32) / 127.0) - 1 # normalize image
  prediction = model.predict(data)
  index = np.argmax(prediction)
  text = f'{labels[index]}:'
  text2 = '{:.3f}'.format(round(prediction[0][index] * 100, 3))

  cv2.putText(img, text, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
  cv2.putText(img, text2, (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
  return list(prediction[0]), index

def main():

  cap = cv2.VideoCapture(0)
  model, labels = read_model(MODEL_PATH, LABELS_PATH)

  while True:
    _, img, = cap.read()
    img2 = cv2.flip(img, 1)
    predictions, index = predict(img2, model, labels)
    cv2.imshow("Image", img2)
    cv2.waitKey(1)

if __name__ == '__main__':
  main()