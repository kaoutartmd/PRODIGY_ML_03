import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from collections import Counter

def load_images_from_folder(folder, label):
    images = []
    labels = []

 #Reading and Processing Images:
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (64, 64)).flatten()  # Resize and flatten the image
            images.append(img)
            labels.append(label)
    return np.array(images), np.array(labels)
# Load Cat and Dog images with correct labels
cat_images, cat_labels = load_images_from_folder('cats', 0)  # 0 for Cat
dog_images, dog_labels = load_images_from_folder('dogs', 1)  # 1 for Dog

# Combine the data
X = np.concatenate((cat_images, dog_images), axis=0)
y = np.concatenate((cat_labels, dog_labels), axis=0)

# Check the balance of the dataset
def print_class_distribution(labels, label_names):
    counts = np.bincount(labels)
    for i, label_name in enumerate(label_names):
        print(f"{label_name}: {counts[i]} samples")
print_class_distribution(y, ['Cat', 'Dog'])

#Split the data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

#Training the  the SVM Model
svm_model = SVC(kernel='linear', C=1.0, random_state=42)
svm_model.fit(X_train, y_train)
#Making Predictions and Evaluate the Model  
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Print classification report
print(classification_report(y_test, y_pred, target_names=['Cat', 'Dog']))

# Print confusion matrix
print(confusion_matrix(y_test, y_pred))
# Function to predict on new images and display the image with prediction
def predict_and_show_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        # Preprocess the image for prediction
        resized_img = cv2.resize(img, (64, 64)).flatten().reshape(1, -1)
        prediction = svm_model.predict(resized_img)
        print(f"Predicted label (raw output): {prediction}")  # Debug output
        label = 'Cat' if prediction == 0 else 'Dog'

        # Show the original image
        plt.imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
        plt.title(f"Prediction: {label}")
        plt.axis('off')  # Turn off axis
        plt.show()
    else:
        print("Invalid image")


# Test the model with a new image and show the result
new_image_path = 'cats\cat.4008.jpg'
predict_and_show_image(new_image_path)