# Airline Sentiment Classification

A Natural Language Processing project developed as part of the **AUEB AI Data Factory – Machine Learning & Data Analysis Bootcamp**.

The project explores and compares three different approaches for classifying airline-related tweets into three sentiment categories:

- Negative
- Neutral
- Positive

The main objective is to examine how different text representations and neural network architectures affect sentiment classification performance.

---

## Project Overview

The project follows a progressive NLP workflow, moving from traditional text representations to sequential neural networks and finally to pretrained Transformer models.

The three approaches are:

1. **TF-IDF + Feed-Forward Artificial Neural Network**
2. **Stacked Bidirectional LSTM with Attention**
3. **Fine-Tuned DistilBERT Transformer**

This progression provides a practical comparison between:

- Sparse statistical text representations
- Learned sequential representations
- Pretrained contextual language representations

---

## 1. TF-IDF + Artificial Neural Network

The first approach represents each tweet using **TF-IDF (Term Frequency–Inverse Document Frequency)** features.

The resulting numerical vectors are passed into a feed-forward Artificial Neural Network for three-class sentiment classification.

### Main Components

- Text preprocessing
- TF-IDF vectorization
- Stratified train / validation / test split
- Feed-forward neural network
- Dropout regularization
- Early stopping
- Multi-class classification

### Key Idea

TF-IDF captures the relative importance of words within the dataset and provides a strong baseline for text classification.

However, it does not explicitly model word order or contextual relationships between words.

### Result

The final revised ANN achieved approximately:

**Test Accuracy: ~80.7%**

---

## 2. Bidirectional LSTM with Attention

The second approach treats each tweet as an ordered sequence rather than a fixed TF-IDF feature vector.

The text is converted into token sequences, padded to a fixed length and passed through an embedding layer before being processed by the recurrent neural network.

### Architecture

The model includes:

- Learned word embeddings
- Two stacked Bidirectional LSTM layers
- Attention mechanism
- Dropout regularization
- Fully connected classification layer
- Early stopping

### Key Idea

The Bidirectional LSTM processes each tweet in both forward and backward directions, allowing the model to capture sequential context.

The attention mechanism learns to assign greater importance to the most informative token positions when producing the final sentiment prediction.

### Training Behaviour

Training and validation loss were monitored across epochs.

The model showed signs of overfitting after the first few epochs, demonstrating the importance of validation monitoring and early stopping.

### Result

The model achieved approximately:

**Test Accuracy: ~77–78%**

Negative sentiment was classified most effectively, while neutral sentiment remained the most challenging category.

The experiment demonstrated that a more complex neural architecture does not automatically guarantee better generalization.

---

## 3. DistilBERT Fine-Tuning

The third approach uses the pretrained Transformer model:

**`distilbert-base-uncased`**

Instead of learning language representations from scratch, DistilBERT starts with pretrained contextual representations and is fine-tuned for the Airline Tweets sentiment classification task.

### Transformer Tokenization

The DistilBERT tokenizer:

- Converts text into subword tokens
- Maps tokens to numerical input IDs
- Adds the special tokens required by the model
- Generates attention masks
- Applies padding and truncation

A maximum sequence length of **64 tokens** is used.

Since airline tweets are short-form text, this provides sufficient capacity while keeping training computationally efficient. The slightly larger limit compared with the word-level RNN representation also provides additional room for subword tokenization.

### Fine-Tuning Strategy

The model is fine-tuned end-to-end using:

- AdamW optimizer
- Learning rate: `2e-5`
- Linear learning-rate scheduler with warm-up
- Gradient clipping
- Maximum of 5 epochs
- Early stopping based on validation loss
- Full Transformer fine-tuning

Full fine-tuning was selected instead of freezing Transformer layers so that the pretrained representations could adapt directly to the airline sentiment classification task.

### Preprocessing Consideration

The preprocessing procedure from the earlier exercises was retained for consistency across the project.

Because the selected model is **DistilBERT-base-uncased**, lowercasing is compatible with the pretrained tokenizer.

However, removing punctuation may discard some sentiment-related information, since punctuation such as exclamation or question marks can carry emotional signal. Preserving more of the original tweet structure could therefore be explored in future experiments.

### Training Behaviour

The best validation loss was:

**0.4558 at Epoch 2**

After this point, training loss continued to decrease while validation loss increased, indicating the beginning of overfitting.

Early stopping prevented unnecessary additional training and restored the best-performing model state.

### Test Results

The final DistilBERT model achieved:

- **Test Loss:** 0.4510
- **Test Accuracy:** 83.06%
- **Weighted F1-score:** 0.83
- **Macro F1-score:** 0.78

Class-level F1-scores:

- **Negative:** 0.90
- **Neutral:** 0.68
- **Positive:** 0.76

DistilBERT achieved the strongest overall performance of the three approaches.

---

## Model Comparison

| Model | Text Representation | Test Accuracy | Main Strength |
|---|---|---:|---|
| Feed-Forward ANN | TF-IDF | ~80.7% | Strong and computationally efficient baseline |
| Bi-LSTM + Attention | Learned word embeddings | ~77–78% | Sequential context and attention |
| DistilBERT | Pretrained contextual representations | **83.06%** | Strong contextual language understanding |

---

## Key Learning Outcomes

### Traditional NLP Features

TF-IDF provides a strong baseline for sentiment classification and demonstrates that relatively simple representations can perform effectively on structured text classification tasks.

### Sequential Deep Learning

The Bi-LSTM experiment demonstrates how recurrent neural networks can model word order and sequential context.

The attention mechanism additionally provides a way for the model to emphasize more informative parts of the input sequence.

The experiment also illustrates that increasing architectural complexity does not necessarily guarantee improved generalization.

### Transfer Learning in NLP

DistilBERT produced the strongest results by leveraging pretrained contextual language representations.

Instead of learning the structure of language solely from the Airline Tweets dataset, the Transformer begins with previously learned linguistic knowledge and adapts it to the specific sentiment classification task through fine-tuning.

This demonstrates the practical value of transfer learning in modern Natural Language Processing.

---

## Repository Structure

```text
airline-sentiment-classification/
│
├── notebooks/
│   ├── 01_airline_sentiment.ipynb
│   ├── 02_airline_rnn.ipynb
│   └── 03_airline_sentiment_transformer.ipynb
│
├── data/
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- PyTorch
- Hugging Face Transformers
- Matplotlib
- Jupyter Notebook

---

## Future Improvements

Potential extensions of the project include:

- Preserving more of the original tweet structure during Transformer preprocessing
- Comparing additional pretrained models such as BERT and RoBERTa
- Exploring class-weighted Transformer training
- Experimenting with frozen Transformer layers and gradual unfreezing
- Hyperparameter optimization
- More detailed error analysis of neutral and positive tweets
- Comparing alternative sequence lengths and batch sizes
- Using pretrained embeddings such as GloVe or fastText for the RNN model
- Deploying the best-performing model through a sentiment prediction API or simple web application

---

## Author

**Marios Kapetanos**

Project developed as part of the **AUEB AI Data Factory – Machine Learning & Data Analysis Bootcamp**.
