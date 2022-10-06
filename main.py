#!/usr/bin/env python3

import cv2
import cvzone

cap = cv2.VideoCapture(0)
model = cvzone.Classifier('models/keras_model.h5', 'models/labels.txt')

while True:
  _, img, = cap.read()
  predictions, index = model.getPrediction(img)

  cv2.imshow("Image", img)
  cv2.waitKey(1)