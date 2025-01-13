# Movie Genre Prediction Based on Plot Summaries

## Overview

This project focuses on building a machine-learning model that can predict the genre of a movie based on its plot summary or other textual information. The goal is to utilize Natural Language Processing (NLP) techniques to extract relevant features from the text and apply various machine learning classifiers to categorize movies into their respective genres.

The model leverages a combination of text vectorization techniques such as **TF-IDF** (Term Frequency-Inverse Document Frequency) and **word embeddings** to convert plot summaries into numerical representations that can be fed into machine learning models. Various classifiers, including **Naive Bayes**, **Logistic Regression**, and **Support Vector Machines (SVM)**, are experimented with to evaluate the accuracy and effectiveness of each model in predicting movie genres.

## Problem Statement

The movie industry produces many films every year, spanning a wide variety of genres, from action to comedy, drama, horror, and science fiction. Often, identifying the genre of a movie can be challenging, especially when there is minimal metadata available about the film. This project aims to solve this problem by automatically predicting the genre of a movie based on its plot summary.

### Use Case

For movie recommendation systems, movie databases, or platforms like IMDb, predicting the genre based on a plot summary can significantly enhance the user's experience. The model can be deployed to suggest appropriate genres or recommend similar movies, ultimately improving the discoverability of films within a streaming platform or movie catalog.

## Approach

### Data Collection

The dataset used in this project consists of a collection of movie plot summaries and their associated genres. The genres are multi-class labels (e.g., Action, Comedy, Drama, etc.). This dataset is preprocessed and cleaned to ensure that the text is ready for analysis.

### Text Preprocessing

The textual data undergoes a series of preprocessing steps to clean and prepare it for modeling. This includes:
- **Tokenization**: Splitting the plot summaries into individual words or tokens.
- **Stopword Removal**: Filtering out common words (e.g., "and", "the") that do not contribute meaningful information.
- **Lowercasing**: Converting all text to lowercase to ensure uniformity.
- **Stemming/Lemmatization**: Reducing words to their root forms to handle different inflections (e.g., "running" to "run").

### Feature Extraction

Once the text data is cleaned, it is transformed into numerical features using text vectorization techniques:
- **TF-IDF**: This method calculates the importance of words in a document relative to a corpus of documents. Words that appear frequently in a document but rarely across all documents are given higher importance.
- **Word Embeddings**: Pre-trained models like Word2Vec or GloVe are used to capture semantic meanings of words by converting them into dense vectors, which are then used to represent the entire plot summary.

### Model Training

Various machine learning classifiers are tested on the transformed textual data:
- **Naive Bayes Classifier**: A probabilistic model based on Bayes' theorem, often effective for text classification tasks.
- **Logistic Regression**: A linear model used for binary or multi-class classification problems.
- **Support Vector Machines (SVM)**: A powerful classifier that aims to find the optimal hyperplane that separates different classes in the feature space.

### Model Evaluation

The models are evaluated based on various metrics, including accuracy, precision, recall, and F1-score. Cross-validation is performed to ensure that the model's performance is reliable and not overfitting to the training data.

## Features

- Predict the genre of a movie based on its plot summary or textual information.
- Flexible with various classification algorithms like Naive Bayes, Logistic Regression, and SVM.
- Customizable text preprocessing and feature extraction pipelines.
- Detailed evaluation metrics to measure model performance.

## Setup and Installation

To set up this project, ensure that the following dependencies are installed:
```bash
pip install numpy pandas scikit-learn nltk spacy
```

### Dataset

The dataset can be obtained from open movie databases like [IMDb](https://www.imdb.com/) or [MovieLens](https://grouplens.org/datasets/movielens/), or it can be manually collected. The dataset should contain at least two columns: one for movie plot summaries and another for the corresponding genres.

### Running the Code

Once the necessary libraries are installed, and the dataset is ready, you can run the model by executing the following command:
```bash
python movie_genre_predictor.py
```

This script will process the dataset, train the models, and output the evaluation results for each classifier.

## Future Improvements

- **Deep Learning**: Incorporating deep learning models such as Recurrent Neural Networks (RNN) or Transformers (e.g., BERT) to improve accuracy by capturing complex textual relationships.
- **Multilingual Support**: Extending the model to predict genres for movies in different languages by incorporating multilingual embeddings or translation pipelines.
- **Real-time Prediction**: Integrating the model into a movie recommendation system where users can input plot summaries of upcoming films and get instant genre predictions.

## Contributing

Contributions to this project are welcome! If you have ideas for improving the model, adding new features, or fixing bugs, feel free to fork the repository and create a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
