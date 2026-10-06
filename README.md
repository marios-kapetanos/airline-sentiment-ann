# Airline Sentiment Classification – ANN & Bi-LSTM with Attention

A Natural Language Processing project developed as part of the **AUEB AI Data Factory – Machine Learning & Data Analysis Bootcamp**.

The project explores two different neural-network approaches for classifying airline-related tweets into **Negative, Neutral and Positive sentiment classes**:

1. **TF-IDF + Feed-Forward Artificial Neural Network**
2. **Embedding + Bidirectional LSTM + Attention**

The second approach extends the original classification pipeline by introducing sequence-aware text representation and an attention mechanism.

## Project Overview

The dataset contains tweets about US airlines, with each tweet labeled as:

- Negative
- Neutral
- Positive

The objective is to preprocess the text and compare different neural-network approaches for multi-class sentiment classification.

## Approach 1 – TF-IDF + Feed-Forward ANN

The first model follows a traditional NLP pipeline:

**Raw Tweets → Text Preprocessing → TF-IDF → Feed-Forward ANN → Sentiment Prediction**

The workflow includes:

- Exploratory Data Analysis
- Text cleaning and preprocessing
- Stratified train / validation / test split
- TF-IDF vectorization
- PyTorch tensors and DataLoaders
- Multi-layer feed-forward neural network
- Dropout regularization
- Early stopping based on validation loss
- Evaluation using multiple classification metrics
- Confusion matrix analysis

This approach provides a strong baseline using fixed numerical representations of the tweet text.

## Approach 2 – Bi-LSTM + Attention

The second model extends the project by using a sequential representation of the tweets instead of TF-IDF.

The pipeline follows:

**Raw Tweets → Text Cleaning → Tokenization → Embedding → Bidirectional LSTM → Attention → Sentiment Prediction**

### Text Processing

Tweets are cleaned by:

- Converting text to lowercase
- Removing URLs
- Removing user mentions
- Removing punctuation while preserving hashtag words
- Normalizing whitespace

The dataset is split using a stratified **70 / 15 / 15 train-validation-test split**.

The vocabulary is created using the training data only, with:

- Minimum token frequency: `2`
- `<PAD>` token: `0`
- `<UNK>` token: `1`
- Maximum sequence length: `50`

Tweets are encoded into integer sequences and padded or truncated to a fixed length before being passed to the neural network.

## RNN Architecture

The sequence model consists of:

- **Embedding layer:** 128-dimensional word representations
- **2-layer Bidirectional LSTM**
- **Hidden size:** 64 units per direction
- **Dropout:** 0.3
- **Masked Attention mechanism**
- Additional dropout before classification
- **Linear output layer:** 3 sentiment classes

The bidirectional LSTM processes each tweet in both forward and backward directions, allowing the model to capture contextual information from both sides of each token.

The attention mechanism then learns to assign greater importance to the most informative parts of each tweet.

## Model Training

The RNN model is trained using:

- **Loss:** CrossEntropyLoss
- **Optimizer:** Adam
- **Learning rate:** 0.001
- **Batch size:** 64
- **Maximum epochs:** 20
- **Early stopping patience:** 3 epochs

Training and validation loss are monitored after each epoch.

The model state associated with the **lowest validation loss is saved and restored**, ensuring that the final evaluation uses the best-performing model rather than simply the model from the final training epoch.

## Performance Evaluation

Both approaches are evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Training and validation loss curves

These metrics provide a more complete view of model performance, particularly because sentiment classes may not be evenly distributed.

## Model Development Strategy

The project demonstrates a progression from a traditional text-classification pipeline to a sequence-based deep-learning architecture.

### Model 1

**TF-IDF → Feed-Forward ANN**

This approach represents each tweet as a fixed TF-IDF feature vector.

### Model 2

**Embedding → Bidirectional LSTM → Attention**

This approach preserves token order and allows the model to learn contextual relationships between words.

Validation-loss monitoring and early stopping were used to control overfitting and identify the best training epoch.

## Technologies Used

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- PyTorch

## Repository Structure

```text
airline-sentiment-ann/
│
├── notebooks/
│   ├── airline_sentiment.ipynb
│   └── airline_rnn.ipynb
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

- Preparing textual data for machine learning and deep learning
- Applying TF-IDF vectorization for text classification
- Creating vocabularies and encoded text sequences
- Using embedding layers to learn word representations
- Building feed-forward neural networks with PyTorch
- Building multi-layer Bidirectional LSTM networks
- Implementing an attention mechanism for sequence classification
- Creating training, validation and testing workflows
- Monitoring validation loss across epochs
- Applying early stopping to reduce overfitting
- Restoring the best-performing model based on validation performance
- Evaluating multi-class classification using Accuracy, Precision, Recall and F1 Score
- Comparing traditional NLP representations with sequence-based neural-network approaches

## Future Improvements

Possible extensions of the project include:

- Compare TF-IDF with pretrained Word2Vec or FastText embeddings
- Use pretrained word embeddings in the recurrent neural network
- Experiment with different LSTM architectures and hidden dimensions
- Explore GRU-based architectures
- Apply more advanced techniques for class imbalance
- Perform systematic hyperparameter optimization
- Compare the neural-network approaches with transformer-based NLP models
