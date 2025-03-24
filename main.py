import cv2
import numpy as np
import os
import pickle
from skimage.feature import graycomatrix, graycoprops
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from imutils import paths
import matplotlib.pyplot as plt

# Constants
IMAGE_SIZE = (150, 150)
CLASSES = ["tomato", "not_tomato"]  # Added "not_tomato" for binary classification
GLCM_PROPERTIES = ['contrast', 'energy', 'homogeneity', 'correlation']

# Feature Extraction Function
def extract_features(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not read image {image_path}")
            return None
            
        image = cv2.resize(image, IMAGE_SIZE)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Color Histogram (HSV)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        
        # Texture Features (GLCM)
        glcm = graycomatrix(gray, distances=[1, 3, 5], angles=[0, np.pi/4, np.pi/2], 
                            levels=256, symmetric=True, normed=True)
        texture_features = []
        for prop in GLCM_PROPERTIES:
            texture_features.extend(graycoprops(glcm, prop).flatten())
        
        # Shape Features (Hu Moments)
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
        moments = cv2.moments(binary)
        hu_moments = cv2.HuMoments(moments).flatten()
        
        # Combine all features
        features = np.hstack([hist, texture_features, hu_moments])
        return features
        
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None

# Load Dataset (Includes "not_tomato" for Binary Classification)
def load_dataset(dataset_path):
    features = []
    labels = []
    
    for label, category in enumerate(CLASSES):  # Loop through both classes
        category_path = os.path.join(dataset_path, category)
        image_paths = list(paths.list_images(category_path))

        for image_path in image_paths:
            feature_vector = extract_features(image_path)
            if feature_vector is not None:
                features.append(feature_vector)
                labels.append(label)  # Assign correct label

    return np.array(features), np.array(labels)

# Model Training and Prediction
def main():
    dataset_path = r"C:\Users\admin\Desktop\plant new\dataset"
    
    print("[INFO] Loading dataset and extracting features...")
    X, y = load_dataset(dataset_path)
    
    if len(set(y)) < 2:
        print("[ERROR] The dataset must have at least two classes. Check your dataset folders.")
        return
    
    # Feature Scaling
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train SVM Classifier
    print("[INFO] Training SVM classifier for Tomato Detection...")
    svm = SVC(kernel="rbf", C=10, gamma=0.001, probability=True)
    svm.fit(X_train, y_train)
    
    # Evaluate model
    accuracy = svm.score(X_test, y_test)
    print(f"\n[INFO] SVM Accuracy: {accuracy:.2f}")

    # Save the model
    print("[INFO] Saving Tomato Detection Model...")
    with open("tomato_classifier.pkl", "wb") as f:
        pickle.dump((svm, scaler, CLASSES), f)
    
    # Test on a new image
    image_path = r"C:\Users\admin\Desktop\plant new\newimg\new1.jpeg" # Change this to your test image path
    predict_image(image_path, svm, scaler)

# Predict a Single Image
def predict_image(image_path, model, scaler):
    feature = extract_features(image_path)
    
    if feature is None:
        print("[ERROR] Could not extract features from the given image.")
        return
    
    feature = scaler.transform([feature])
    proba = model.predict_proba(feature)[0]
    prediction = CLASSES[np.argmax(proba)]

    print(f"[RESULT] The image is predicted as: {prediction} ({max(proba):.2f} confidence)")

    # Show Image
    image = cv2.imread(image_path)
    image = cv2.resize(image, (200, 200))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(f"Prediction: {prediction} ({max(proba):.2f})")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()
