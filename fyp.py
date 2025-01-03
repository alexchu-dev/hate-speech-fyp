import re
import emoji
import unicodedata
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import pandas as pd
import torch

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
    # Normalize Unicode characters (e.g., ＦＵＣＫ → FUCK)
    text = unicodedata.normalize('NFKC', text)
    
    # Convert emojis to descriptive text
    text = emoji.demojize(text, delimiters=(" ", " "))
    
    # Remove special symbols (e.g., @#$%^)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Example usage
text = "😂🖕 ＦＵＣＫ you"
processed_text = preprocess_text(text)
print(processed_text)

# Load dataset
data = pd.read_csv("datasets/HatemojiBuild-train.csv")
print(data.head())
 
# Preprocess text
data["text"] = data["text"].apply(preprocess_text)

# Tokenize data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
encodings = tokenizer(list(data["text"]), truncation=True, padding=True, max_length=128)
labels = torch.tensor(data["label_gold"].values)

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
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])

# Load BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)  # 3 labels: Hate, Offensive, Neutral

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
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    processing_class=tokenizer,
)

# Train the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./fyp/bert_cyberhate_model")
tokenizer.save_pretrained("./fyp/bert_cyberhate_model")