# 1) Movie Genre Prediction Based on Plot Summaries
---
#### Overview  
A machine-learning model predicts movie genres from plot summaries using NLP techniques. Features are extracted with **TF-IDF** and **word embeddings**, and classifiers like **Naive Bayes**, **Logistic Regression**, and **SVM** categorize movies by genre.  

#### Key Highlights  
- **Problem**: Predict genres using minimal metadata (plot summaries).  
- **Approach**: Text preprocessing, feature extraction, and classification.  
- **Features**: Supports multiple classifiers, customizable preprocessing, and evaluation metrics.  

#### Installation & Usage  
Install dependencies:  
```bash
pip install numpy pandas scikit-learn nltk spacy
```  
Run the model:  
```bash
python movie_genre_predictor.py
```  

#### Future Enhancements  
- Integrate deep learning models (e.g., RNN, BERT).  
- Support for multilingual data.  
- Real-time predictions for recommendation systems.

## Future Improvements

- Incorporating deep learning models (e.g., RNN, BERT).
- Adding multilingual support.
- Real-time genre predictions in recommendation systems.

# 2) Fraud Detection Model

This project implements a machine learning-based system for detecting fraudulent transactions in financial datasets. The goal is to classify transactions as either **fraudulent** or **legitimate** using historical data. By applying various machine learning algorithms, this model can help financial institutions detect fraud more effectively and efficiently.

---

## Project Overview

The project follows a comprehensive approach to handle raw transaction data and build a fraud detection system with the following key steps:

1. **Data Preprocessing**: 
    - The dataset is cleaned by removing rows with missing values.
    - Timestamp columns are processed to extract useful features like year, month, day, and hour.
    - Categorical variables are encoded using label encoding, making the data suitable for machine learning models.
  
2. **Feature Engineering**: 
    - Timestamp features are extracted from the transaction time to capture patterns based on the time of day, month, and year.
    - The target variable, `is_fraud`, is used to train the model to predict fraudulent transactions.
  
3. **Modeling**: 
    - **Logistic Regression**: A linear model that is trained and used to classify transactions based on their features.
    - **Decision Tree Classifier**: A non-linear model that builds decision paths based on transaction features.
    - **Random Forest Classifier**: An ensemble model that uses multiple decision trees to provide a robust prediction.

4. **Model Evaluation**: 
    - The models' performance is assessed using metrics such as **accuracy**, **confusion matrix**, and **classification report**.
    - The classification report includes precision, recall, f1-score, and support, which are essential for evaluating the model's effectiveness in detecting fraud.

5. **Results**: 
    - The models are compared to determine which performs best in detecting fraudulent transactions, helping identify the most reliable algorithm for this task.

## Requirements

To run the project, ensure the following Python libraries are installed:

- `pandas` – for data manipulation and analysis.
- `scikit-learn` – for machine learning models and evaluation.
- `numpy` – for numerical operations.
  
You can install them using the following command:

```bash
pip install pandas scikit-learn numpy
```

## How to Run

1. Download the training and test datasets and update the paths in the script.
2. Ensure the data is preprocessed by cleaning and encoding as specified.
3. Run the Python script to train the models and generate performance metrics.
4. The results will include the accuracy, confusion matrix, and classification report for each model.

## Conclusion

This project demonstrates a basic approach to fraud detection using machine learning models. By leveraging the strengths of logistic regression, decision trees, and random forests, we can identify fraudulent transactions with reasonable accuracy. Further optimizations and model tuning could improve the system's performance in real-world scenarios.


# 3)SMS Spam Classification

This project is designed to classify SMS messages as either spam or legitimate (ham) using Natural Language Processing (NLP) and machine learning techniques. The model helps identify unwanted or fraudulent messages, making it an effective tool for spam detection.

---

### Key Features:

- **Dataset Preparation**: 
  - Processes a labeled dataset of SMS messages containing "spam" and "ham" labels.
  - Handles data cleaning and preprocessing to prepare it for model training.

- **Text Preprocessing**:
  - Converts SMS text to lowercase for uniformity.
  - Removes unnecessary characters, punctuation, and numbers.
  - Prepares text data for numerical representation.

- **Feature Extraction**:
  - Uses TF-IDF (Term Frequency-Inverse Document Frequency) vectorization to convert text into numerical features that capture the significance of words in each message.

- **Model Training**:
  - Implements a **Multinomial Naive Bayes Classifier**, a widely used algorithm for text classification tasks.
  - Trains the model on preprocessed and vectorized text data.

- **Evaluation Metrics**:
  - Assesses model performance using:
    - **Accuracy**: Measures overall prediction correctness.
    - **Confusion Matrix**: Breaks down true/false positives and negatives.
    - **Classification Report**: Provides precision, recall, and F1-score for spam and ham classes.

- **Real-Time Prediction**:
  - Allows users to input custom SMS messages and predicts whether they are spam or legitimate.

---

### Technologies Used:

- **Python Libraries**:
  - Pandas: For data manipulation and preprocessing.
  - Scikit-learn: For text vectorization, model training, and evaluation.

- **Machine Learning Model**:
  - Multinomial Naive Bayes for classification.

- **Text Processing**:
  - TF-IDF Vectorization for transforming text data into meaningful numerical features.

---

### Project Highlights:

This project demonstrates how to:
- Clean and preprocess textual data for NLP tasks.
- Extract meaningful features from text using vectorization techniques.
- Train and evaluate a machine learning model for effective spam detection.
- Implement a simple and user-friendly real-time prediction system for spam classification.

This solution is a step towards creating automated systems to filter and manage spam messages effectively.


### Contributing

Feel free to fork the repository and create pull requests for improvements!



