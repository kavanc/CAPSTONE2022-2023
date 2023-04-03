'''
    Imports the csv file with all of the data generated,
    trains a series of models, and tests them to determine the effectiveness
'''

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
from create_csv import create_csv
import pathlib
from create_csv import get_image_list


# potential models to be trained and evaluated
PIPELINES = {
    'rf': make_pipeline(StandardScaler(), RandomForestClassifier()),
}

def generate_models(X_train, y_train) -> dict:
    models = {}
    for mt, pipeline in PIPELINES.items():
        print(f"Training model {mt}")
        models[mt] = pipeline.fit(X_train, y_train)
        print(f"Model {mt} trained\n")


    return models

def write_model(model):
    with open(f"./models/body_language.pkl", "wb") as f:
        pickle.dump(model,f)


if __name__ == '__main__':
    c_path = pathlib.Path().resolve()
    img_list = get_image_list(c_path)
    img_list = [x for x in img_list if "evaluation_data" not in x] # comment out this line if evaluation data is to be included

    create_csv(f"{c_path}/models/coords.csv", img_list)

    # read in data
    df = pd.read_csv('./models/coords.csv')

    X = df.drop('class', axis=1) # features
    y = df['class'] # target value

    print("Splitting Samples...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)
    print("Samples split successfully!\n")

    # train the models
    print("Training models...")
    models = generate_models(X_train, y_train)

    print("\nModel training complete!\n")

    # tests the models on the test set
    print("Generating Accuracy metrics:")
    fit_models = {}
    for algo, pipeline in PIPELINES.items():
        model = pipeline.fit(X_train, y_train)
        fit_models[algo] = model

    for algo, model in fit_models.items():
        yhat = model.predict(X_test)
        print(algo, "Accuracy: ", accuracy_score(y_test, yhat))


    # writes the models to pickle files for later use
    print("Writing Models to pickle files")
    write_model(models['rf'])

