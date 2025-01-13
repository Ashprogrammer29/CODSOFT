## 1)Movie Genre Prediction Based on Plot Summaries

## Overview

This project builds a machine-learning model to predict movie genres from plot summaries using Natural Language Processing (NLP) techniques. It transforms textual data using **TF-IDF** and **word embeddings**, and tests various classifiers, including **Naive Bayes**, **Logistic Regression**, and **SVM**, to categorize movies by genre.

## Problem Statement

Identifying movie genres can be challenging, especially with minimal metadata. This project solves the problem by predicting a movie's genre based on its plot summary, enabling better recommendations and movie discovery on platforms like IMDb.

## Approach

### Data Collection & Parsing

The dataset includes movie plot summaries with associated genres. The text is parsed to create structured data and handle missing genres in test data by assigning a default value like "unknown".

### Text Preprocessing

The plot summaries are cleaned by:
- Tokenizing
- Removing stopwords
- Converting text to lowercase
- Stemming or Lemmatization

### Feature Extraction

Text data is transformed using **TF-IDF** and **word embeddings** to create numerical features suitable for machine learning models.

### Model Training

Classifiers like **Naive Bayes**, **Logistic Regression**, and **SVM** are trained on the features to predict genres.

### Evaluation

Models are evaluated using accuracy, precision, recall, and F1-score, with cross-validation to prevent overfitting.

## Features

- Predict movie genres based on plot summaries.
- Use multiple classification algorithms.
- Customizable preprocessing and feature extraction.
- Detailed evaluation metrics.

## Setup and Installation

Install the required dependencies:

```bash
pip install numpy pandas scikit-learn nltk spacy
```

### Dataset

You can use movie datasets like those from [IMDb](https://www.imdb.com/) or [MovieLens](https://grouplens.org/datasets/movielens/).

### Running the Code

Run the model with:

```bash
python movie_genre_predictor.py
```

## Future Improvements

- Incorporating deep learning models (e.g., RNN, BERT).
- Adding multilingual support.
- Real-time genre predictions in recommendation systems.

## Contributing

Feel free to fork the repository and create pull requests for improvements!



