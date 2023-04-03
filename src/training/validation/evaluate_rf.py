import pathlib
import cv2
import numpy as np
import csv
import os
import mediapipe as mp
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# generates complete paths for all jpg images in the training folder
def get_image_list(path: str) -> list[str]:
    imgs = []

    for dirpath, _, filenames in os.walk(path):
        for filename in filenames:
            if filename.lower().endswith('.jpg'):
                imgs.append(os.path.join(dirpath, filename))

    return imgs

# builds a csv file of training data
def create_csv(out_path: str = './', img_list: list[str] = None):
    mp_holistic = mp.solutions.holistic
    results = None
    first_loop = True
    s_count = 0
    f_count = 0

    print("Writing to CSV...")
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for img_path in img_list:

            img_class = img_path.split('\\')[-1].split('_')[0].lower() # works on windows only
            if img_class == "average":
                img_class = "Normal"

            if img_class == "fighting":
                img_class = "Fighting"
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            results = holistic.process(image)

            # Adds the column names to the csv file
            # Done this way because depending on the image set
            # the number of columns is variable
            if first_loop:
                # generate header
                num_coords = len(results.pose_landmarks.landmark)
                landmarks = ['class']
                for i in range(1, num_coords + 1):
                    landmarks += [f"x{i}", f"y{i}", f"z{i}", f"v{i}"]

                # write to file
                with open(out_path, mode='w', newline='') as f:
                    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    csv_writer.writerow(landmarks)

                first_loop = False

            # Extract pose landmarks from images and write to csv
            try:
                # Extract Pose landmarks
                pose = results.pose_landmarks.landmark
                row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
                
                # Append class name 
                row.insert(0, img_class)
                
                # Export to CSV
                with open(out_path, mode='a', newline='') as f:
                    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    csv_writer.writerow(row) 

                s_count += 1
            except:
                f_count += 1
        

    print(f"{s_count} success")
    print(f"{f_count} failed")

    print("Writing to CSV complete!")

def get_test_vals():
    c_path = pathlib.Path().resolve()
    img_list = get_image_list(f"{c_path}")
    create_csv(f"{c_path}/coords.csv", img_list)

    df = pd.read_csv('./coords.csv')

    X = df.drop('class', axis=1)
    y = df['class']

    # use most of the data as test data
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

    return X_test, y_test

# tests the predict model for accuracy
def test_accuracy(model, x, y) -> float:
    yhat = model.predict(x)
    return accuracy_score(y, yhat)

def test_precision(model, x, y) -> float:
    yhat = model.predict(x)
    return precision_score(y, yhat, average='micro')

def test_recall(model, x, y) -> float:
    yhat = model.predict(x)
    return recall_score(y, yhat, average='micro')

def test_f1(model, x, y) -> float:
    yhat = model.predict(x)
    return f1_score(y, yhat, average='micro')

# prints results
def evaluate_models(x, y):
    rf_model = None
    with open('../models/body_language.pkl', "rb") as f:
        rf_model = pickle.load(f)

    print("Accuracy:")
    rf_acc = test_accuracy(rf_model, x, y)
    print(rf_acc)

    print("\nPrecision")
    print(test_precision(rf_model, x, y))

    print("\nRecall:")
    print(test_recall(rf_model, x, y))

    print("\nF1:")
    print(test_f1(rf_model, x, y))

if __name__ == '__main__':
    x, y = get_test_vals()
    evaluate_models(x, y)