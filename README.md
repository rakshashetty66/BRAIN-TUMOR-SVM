# Brain Tumor Detection and Classification Using Support Vector Machine (SVM)

# Project Overview
This project aims to detect and classify brain tumors using MRI images. We use Support Vector Machine (SVM) for classification, a machine learning algorithm particularly effective for binary classification problems. This project involves image preprocessing, feature extraction, and SVM model training to classify MRI images as either tumor-positive or tumor-negative.
The entire process is implemented in a Jupyter Notebook for ease of explanation, reproducibility, and visualization.

# Steps
1. Install Required Libraries
Ensure you have the following Python libraries installed:
pip install numpy pandas scikit-learn matplotlib seaborn opencv-python pillow

2. Dataset Acquisition
We used an open-source MRI brain tumor dataset consisting of labeled MRI scans (tumor or no tumor).
You can download the dataset from Kaggle or use your preferred dataset. Ensure your dataset has both tumor-positive and tumor-negative MRI images.

4. Load and Explore Dataset
Use Pandas to load metadata (if available) and OpenCV or Pillow to load the MRI images.
Explore the data to understand the structure, distribution of classes, and visualize a few sample images.

4. Preprocess the Data
Resize Images: To ensure uniformity, resize all MRI images to a fixed size, e.g., 128x128 pixels.
Convert to Grayscale: Convert the images to grayscale, as color information is not necessary for MRI classification.
Normalize the Pixel Values: Scale the pixel values to a range of [0, 1] by dividing them by 255.
Flatten Images: Convert the 2D images into 1D arrays for input into the SVM classifier.

5. Feature Extraction
Extract relevant features from the preprocessed images. For simplicity, in this project, we use pixel intensities as features by flattening the 2D image into a 1D vector.

6. Split the Dataset
Split the data into training and test sets, typically using an 80/20 ratio.

7. Train the SVM Classifier
Support Vector Machine (SVM) is used as the classifier.
Use Scikit-learn's SVC class to train the model on the training dataset.

8. Model Evaluation
Evaluate the model using the test set.
Calculate accuracy, precision, recall, and F1-score using Scikit-learn's metrics.

9. Visualize Results
Visualize some predictions alongside the ground truth to understand how well the model is performing.

10. Save and Export the Model
Save the trained SVM model using joblib or pickle so it can be reused later without retraining.

11. Conclusion and Future Work
The current model uses basic pixel intensities for feature extraction. For better performance, you can explore more advanced techniques such as:
Feature Engineering: Use image processing techniques like edge detection or histogram of gradients (HOG).
Deep Learning: Consider using CNNs (Convolutional Neural Networks) for more accurate results.
Cross-validation: Perform cross-validation to fine-tune hyperparameters for better performance.

# Requirements
1. Python 3.x
2. Jupyter Notebook
3. Numpy, Pandas, Scikit-learn, Matplotlib, Seaborn, OpenCV, Pillow
4. Use Scikit-learn's train_test_split to perform this split.
