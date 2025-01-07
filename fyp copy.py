import re
import emoji
import unicodedata
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

# Declaring device
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("###Using MPS on macOS.###\n")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("###Using CUDA.###\n")
else:
    if not torch.backends.mps.is_built():
        print("current PyTorch install was not built with MPS enabled.")
    else:
        print("current MacOS version is not 12.3+ and/or you do not have an MPS-enabled device on this machine.")
    device = torch.device("cpu")
    print("###Using CPU.###\n")

def preprocess_text(text):
    text = unicodedata.normalize('NFKC', text)
    text = emoji.demojize(text, delimiters=(" ", " "))
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text.replace('_', ' '))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Load dataset
data = pd.read_csv("datasets\labeled_data_added_emoji.csv")
print("Original dataset shape:", data.shape)
# Preprocess text
data["text"] = data["text"].apply(preprocess_text)
print("Preprocessed dataset shape:", data.shape)
print("Preprocessed dataset label counts:", data["label"].value_counts())

# Tokenize data
tokenizer = BertTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
encodings = tokenizer(list(data["text"]), truncation=True, padding=True, max_length=128)
labels = torch.tensor(data["label"].values)
# Create PyTorch Dataset
class HateSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

dataset = HateSpeechDataset(encodings, labels)
# Split dataset
train_size = int(0.8 * len(dataset))
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size - 1000, 1000])
print("Train size:", len(train_dataset))
print("Val size:", len(val_dataset))
print("Test size:", len(test_dataset))
print("Train labels:", train_dataset.dataset.labels[train_dataset.indices].unique(return_counts=True))

# Load BERT model
model = BertForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
# model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)  # 3 labels: Hate, Offensive, Neutral

# Training arguments
training_args = TrainingArguments(
    output_dir='./fyp/results',
    num_train_epochs=50,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir='./fyp/logs',
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

# Define evaluation metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

predictions = trainer.predict(test_dataset)
logits = predictions.predictions
true_labels = predictions.label_ids
predicted_labels = np.argmax(logits, axis=-1)
report = classification_report(true_labels, predicted_labels, target_names=["Class 0", "Class 1"])
print("Classification Report:")
print(report)

# Save the fine-tuned model
model.save_pretrained("./fyp/bert_cyberhate_model")
tokenizer.save_pretrained("./fyp/bert_cyberhate_model")

torch.cuda.empty_cache()