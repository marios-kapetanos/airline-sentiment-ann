# Airline Sentiment Classification with ANN

A machine learning project developed as part of the **AUEB AI Data Factory – Machine Learning & Data Analysis Bootcamp**.

The project focuses on classifying airline-related tweets into **Negative, Neutral and Positive sentiment classes** using text preprocessing, TF-IDF vectorization and a feed-forward Artificial Neural Network implemented with PyTorch.

## Project Overview

The dataset contains tweets about US airlines, with each tweet labeled as:

- Negative
- Neutral
- Positive

The objective is to preprocess the text, convert tweets into numerical representations and train a neural network to predict sentiment accurately.

## Methodology

The project includes:

- Exploratory data analysis
- Text cleaning and preprocessing
- Token handling and normalization
- TF-IDF vectorization
- Train / validation / test split
- PyTorch tensors and DataLoaders
- Feed-forward neural network design
- Early stopping
- Validation loss monitoring
- Model evaluation
- Confusion matrix analysis
- Model revision based on validation performance

## Technologies Used

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- PyTorch

## Model Training

The model is trained as a multi-class classifier for the three sentiment categories.

Training performance is evaluated using validation loss across epochs, while early stopping is used to reduce the risk of overfitting.

The project also includes a revised training approach that restores the model state associated with the best validation performance.

## Performance Evaluation

Model performance is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Training and validation loss curves

These metrics provide a more complete view of performance than accuracy alone, especially when class distributions are imbalanced.

## Repository Structure

```text
airline-sentiment-ann/
│
├── notebooks/
│   └── airline_sentiment.ipynb
│
├── data/
│   └── Tweets.csv
│
├── README.md
├── requirements.txt
└── .gitignore
```

## Key Learning Outcomes

Through this project I gained hands-on experience with:

- Preparing textual data for machine learning
- Converting text into numerical features with TF-IDF
- Building a multi-layer neural network with PyTorch
- Creating training and validation workflows
- Monitoring validation loss across epochs
- Applying early stopping to reduce overfitting
- Evaluating multi-class classification using multiple metrics
- Using validation results to revise and improve the training process

## Future Improvements

Possible extensions of the project include:

- Compare TF-IDF with Word2Vec or FastText embeddings
- Experiment with deeper neural network architectures
- Explore alternative NLP models
- Apply more advanced class-imbalance techniques
- Perform systematic hyperparameter optimization
- Compare the ANN with traditional classifiers such as Logistic Regression or SVM
