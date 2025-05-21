# Movie Reviews Sentiment Analysis

## Overview

This project implements sentiment analysis on movie reviews using both traditional Machine Learning and BERT-based approaches. The system can classify movie reviews as either positive (1) or negative (0) with high accuracy.

## Project Structure

```
├── BERT accuracy/          # BERT model performance visualizations
├── ML models accuracy/     # ML models performance visualizations
├── NLP_Data/               # Dataset files
│   └── all_reviews.csv     # Combined dataset
├── NLP project.py          # Main project implementation
├── NLP project.ipynb       # Main project implementation as Jupyter Notebook
├── Documentation.docx      # Detailed documentation
└── requirements.txt        # Project dependencies
```

## Setup and Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd <repository-name>
```

2. Create a virtual environment (recommended):

```bash
python -m venv venv
.\venv\Scripts\Activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Download NLTK resources:
   The script will automatically download required NLTK resources on first run, or you can manually download them:

```python
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords', 'averaged_perceptron_tagger'])
```

## Model Training

Run the main script to train both ML models and BERT:

```bash
python "NLP project.py"
```

## Models

### Traditional Machine Learning

- Logistic Regression
- Linear SVC
- Random Forest
  All models use TF-IDF vectorization with unigrams and bigrams.

### BERT Model

- Base: textattack/bert-base-uncased-SST-2
- Fine-tuned for sentiment analysis
- Includes early stopping and model checkpointing

## Performance

### Machine Learning Models

- Results available in [`ML models accuracy/`](./ML%20models%20accuracy/)
- Classification reports for each model
- Comparative performance analysis

### BERT Model

- Results available in [`BERT accuracy/`](./BERT%20accuracy/)
- Training loss curves
- Evaluation metrics
- Final test results

## Documentation

For detailed information about:

- Data preprocessing steps
- Model architectures
- Training configurations
- Performance metrics
- Implementation details

Please refer to [`Documentation.docx`](./Documentation.docx) for more details.

## Usage

To use the trained model for predictions (after training and saving the model):

```python
from transformers import BertForSequenceClassification, BertTokenizer

# Load the model
model_path = "./bert_model"
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

# Prepare text
text = "Your movie review here"
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)

# Get prediction
outputs = model(**inputs)
prediction = torch.argmax(outputs.logits, dim=-1)
sentiment = "positive" if prediction == 1 else "negative"
```

