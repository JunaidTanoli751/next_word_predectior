# Next Word Predictor using LSTM

A Deep Learning project that predicts the next word in a sentence using Long Short-Term Memory (LSTM) neural networks. The model is trained on conversational human chat data to generate contextually relevant word predictions.

## 🚀 Key Features

- **LSTM-based Architecture**: Utilizes recurrent neural networks for sequence prediction
- **Word Embeddings**: 50-dimensional word embeddings for semantic representation
- **Sequence Padding**: Manages variable-length input sequences
- **Performance**: Achieves approximately 51% training accuracy on conversational data
- **Text Preprocessing**: Comprehensive data cleaning and tokenization
- **Model Persistence**: Saves trained model and tokenizer for future use

## 📊 Model Architecture

```
Model: Sequential
_________________________________________________________________
Layer (type)                Output Shape              Param #
=================================================================
Embedding (Embedding)       (None, 150, 50)          139,150
LSTM (LSTM)                 (None, 150)              120,600
Dense (Dense)               (None, 2783)             420,233
=================================================================
Total params: 679,983 (2.59 MB)
Trainable params: 679,983 (2.59 MB)
Non-trainable params: 0 (0.00 B)
```

### Architecture Components:
- **Embedding Layer**: Transforms words into dense vectors (vocab_size=2783, embedding_dim=50)
- **LSTM Layer**: Handles sequential data with 150 units
- **Dense Layer**: Generates probability distribution over vocabulary (softmax activation)

## 📁 Project Structure

```
next-word-predictor-lstm/
├── human_chat.txt              # Training dataset (conversational text)
├── next_word_predictor.ipynb   # Main Jupyter notebook
├── lstm_model.h5               # Trained model (HDF5 format)
├── lstm_model.keras            # Trained model (Keras format)
├── tokenizer.pkl               # Fitted tokenizer object
├── max_len.txt                 # Maximum sequence length
└── README.md                   # Project documentation
```

## 🛠️ Installation

### Prerequisites
```bash
Python 3.7+
TensorFlow 2.x
NumPy
```

### Setup
```bash
# Clone the repository
git clone https://github.com/adeel-iqbal/next-word-predictor-lstm.git
cd next-word-predictor-lstm

# Install required packages
pip install tensorflow numpy
```

## 💻 Usage

### Training the Model

```python
import warnings
warnings.filterwarnings('ignore')

# Load and preprocess data
with open('human_chat.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()

# Clean text data
import re
cleaned_lines = []
for line in lines:
    line = re.sub(r'Human\s*\d*:\s*', '', line)
    line = line.strip().lower()
    line = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', line)
    if line:
        cleaned_lines.append(line)

# Train model (see notebook for complete code)
# model.fit(X, y, epochs=20, batch_size=32)
```

### Making Predictions

```python
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load saved model and tokenizer
model = load_model('lstm_model.keras')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
with open('max_len.txt', 'r') as f:
    max_len = int(f.read())

# Prediction function
def predict_next_word(model, tokenizer, max_len, seed_text, num_words=1):
    for _ in range(num_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_len-1, padding='pre')
        predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1)[0]
        
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                seed_text += " " + word
                break
    return seed_text

# Example usage
result = predict_next_word(model, tokenizer, max_len, "what is your", num_words=4)
print(result)  # Output: "what is your day going to be"
```

## 📈 Model Performance

| Metric | Value |
|--------|-------|
| Final Training Loss | 2.2417 |
| Final Training Accuracy | 50.89% |
| Vocabulary Size | 2,783 words |
| Max Sequence Length | 151 tokens |
| Training Epochs | 20 |
| Batch Size | 32 |

### Training Progress
- **Epoch 1**: Loss: 6.48, Accuracy: 3.68%
- **Epoch 10**: Loss: 3.98, Accuracy: 21.01%
- **Epoch 20**: Loss: 2.24, Accuracy: 50.89%

## 🧪 Example Predictions

```python
# Input: "what is your"
# Output: "what is your day going to be"

# Input: "how are"
# Output: "how are you"
```

## 📝 Data Preprocessing

The preprocessing pipeline consists of:

1. **Identifier Removal**: Remove "Human 1:", "Human 2:" labels using regex
2. **Normalization**: Transform text to lowercase
3. **Cleaning**: Eliminate special characters (retain alphanumeric and basic punctuation)
4. **Tokenization**: Transform words into integer sequences
5. **N-gram Generation**: Build training sequences from conversations
6. **Padding**: Standardize sequence lengths to 150 tokens

## 🔧 Technologies Used

- **TensorFlow/Keras**: Deep learning framework
- **NumPy**: Numerical computations
- **Python**: Programming language
- **LSTM**: Recurrent neural network architecture
- **Word Embeddings**: Dense vector representations

## 🙏 Acknowledgments

- Dataset: Human conversational chat data
- Framework: TensorFlow/Keras
- Inspiration: Natural Language Processing research

---

⭐ **Star this repository if you find it helpful!**
