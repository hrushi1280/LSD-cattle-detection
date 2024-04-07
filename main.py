from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import RandomOverSampler
import os
import cv2
import numpy as np
X = []
y = []
image_size = 150
healthy_folder = '/content/drive/MyDrive/cows dataset-20240407T163325Z-001/cows dataset/train/Healthy cows'
diseased_folder = '/content/drive/MyDrive/cows dataset-20240407T163325Z-001/cows dataset/train/Lumpy cows'
for img_name in os.listdir(healthy_folder):
    img = cv2.imread(os.path.join(healthy_folder, img_name))
    if image_size:
        img = cv2.resize(img, (image_size, image_size))
    X.append(img.flatten())  # Flatten the image
    y.append(0)  # 0 for healthy
    for img_name in os.listdir(diseased_folder):
    img = cv2.imread(os.path.join(diseased_folder, img_name))
    if image_size:
        img = cv2.resize(img, (image_size, image_size))
    X.append(img.flatten())  # Flatten the image
    y.append(1)  # 1 for diseased
    X = np.array(X)
y = np.array(y)

# Shuffle the data
X, y = shuffle(X, y, random_state=101)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)
X_train_resampled, y_train_resampled = RandomOverSampler().fit_resample(X_train, y_train)
model = RandomForestClassifier(random_state=101)
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
}
grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train_resampled, y_train_resampled)
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Evaluate accuracy and other metrics
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Load the image
img = cv2.imread('/content/simple.jpg')

# Resize the image to the same size as the training images
if image_size:
    img = cv2.resize(img, (image_size, image_size))

# Flatten the image
img = img.flatten()

# Predict the class of the image
y_pred = best_model.predict([img])

# Print the prediction
print(y_pred)
if y_pred == 0:
  print("Cow is healthy")
else:
  print("Cow is Affected")
