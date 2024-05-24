from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import shuffle
from imblearn.over_sampling import RandomOverSampler
import os
from flask import Flask, request, redirect, url_for, render_template_string
import os
from PIL import Image
import numpy as np

app = Flask(__name__)

# Set the upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def upload_form():
    return render_template_string('''
    <!doctype html>
    <style> 
    body {
        background-image: url('/static/cover.png');
        background-size: 50%; /* Cover the entire page */
        # background-position: center; /* Center the image */
        background-repeat: repeat; /* Do not repeat the image */
    }
    .parent {
        width: 100%;
        display: flex;
        justify-content: center;
    }
    .child {
        width: 100%;
        height: 100%;
        color: red;
        text-align: center;
    }
    </style>
    <title>Welcome to Cattle detection system</title>
    <div class="parent">
  <div class="child">
    <h1>Welcome to Cattle detection system</h1>
    <h2>Upload a File</h2>
    <form method=post enctype=multipart/form-data action="/upload">
      <input type=file name=file>
      <input type=submit value=Upload>
    </form></div>
</div>
    ''')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        X = []
        y = []
        image_size = 150
        healthy_folder = '/Volumes/Code/LSD-cattle-detection/train/Healthy cows'
        diseased_folder = '/Volumes/Code/LSD-cattle-detection/train/Lumpy cows'
        # Load healthy images
        for img_name in os.listdir(healthy_folder):
            img_path = os.path.join(healthy_folder, img_name)
            img = Image.open(img_path).convert('RGB')
            if image_size:
                img = img.resize((image_size, image_size))
            img_array = np.array(img).flatten()
            X.append(img_array)  # Flatten the image
            y.append(0)  # 0 for healthy

        # Load diseased images
        for img_name in os.listdir(diseased_folder):
            img_path = os.path.join(diseased_folder, img_name)
            img = Image.open(img_path).convert('RGB')
            if image_size:
                img = img.resize((image_size, image_size))
            img_array = np.array(img).flatten()
            X.append(img_array)  # Flatten the image
            y.append(1)  # 1 for diseased

        X = np.array(X)
        y = np.array(y)

        # Shuffle the data
        X, y = shuffle(X, y, random_state=101)

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)

        # Resampling
        X_train_resampled, y_train_resampled = RandomOverSampler().fit_resample(X_train, y_train)

        # Model training
        model = RandomForestClassifier(random_state=101)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
        }
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy')
        grid_search.fit(X_train_resampled, y_train_resampled)
        best_model = grid_search.best_estimator_

        # Prediction on test set
        y_pred = best_model.predict(X_test)

        # Evaluation
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy}")
        print("Classification Report:\n", classification_report(y_test, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

        # Predict the class of the single image
        img = Image.open(file).convert('RGB')
        if image_size:
            img = img.resize((image_size, image_size))
        img_array = np.array(img).flatten()
        y_pred = best_model.predict([img_array])

        # Print the prediction
        print(y_pred)
        if y_pred == 0:
            return '\033[1mCow is healthy\033[0m'

        else:
            return '\033[1mCow is affected\033[0m'

if __name__ == "__main__":
    app.run(debug=True)

