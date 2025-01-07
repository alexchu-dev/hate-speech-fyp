import numpy as np
import pandas as pd
import torch
import re
import emoji
import unicodedata
from tqdm import tqdm
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from transformers import AdamW
from transformers import get_scheduler
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score

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
# Load dataset
df_train = pd.read_csv("datasets/HatemojiBuild-train.csv")
df_test = pd.read_csv("datasets/HatemojiBuild-test.csv")
df_val = pd.read_csv("datasets/HatemojiBuild-validation.csv")
print(df_train.head())

# Preprocess data
train_texts, train_labels = df_train['text'], df_train['label']
val_texts, val_labels = df_val['text'], df_val['label']
test_texts, test_labels = df_test['text'], df_test['label']

print("Train size:", len(train_texts))
print("Val size:", len(val_texts))
print("Test size:", len(test_texts))

def preprocess_text(text):
    text = unicodedata.normalize('NFKC', text)
    text = emoji.demojize(text, delimiters=(" ", " "))
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text.replace('_', ' '))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Tokenize data
tokenizer = BertTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

def tokenize_text(texts, labels, max_length=128):
    input_ids = []
    attention_masks = []
    
    for text in texts:
        encoded = tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        input_ids.append(encoded['input_ids'].squeeze(0))
        attention_masks.append(encoded['attention_mask'].squeeze(0))
        
    input_ids = torch.stack(input_ids)
    attention_masks = torch.stack(attention_masks)
    labels = torch.tensor(labels.values)
    
    return {"input_ids": input_ids, "attention_mask": attention_masks, "labels": labels}

train_encodings = tokenize_text(train_texts, train_labels)
val_encodings = tokenize_text(val_texts, val_labels)
test_encodings = tokenize_text(test_texts, test_labels)

train_labels = torch.tensor(train_labels.values)
val_labels = torch.tensor(val_labels.values)
test_labels = torch.tensor(test_labels.values)

class EmojiDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)
    
train_dataset = EmojiDataset(train_encodings, train_labels)
val_dataset = EmojiDataset(val_encodings, val_labels)
test_dataset = EmojiDataset(test_encodings, test_labels)

# Model architecture
model = BertForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment", num_labels=2)

# Freeze all parameters
for param in model.base_model.parameters():
    param.requires_grad = False
    
# Unfreeze classification head
for param in model.classifier.parameters():
    param.requires_grad = True

model.to(device)

optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)

# Train the model
batch_size = 16
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

num_epochs = 50

num_training_steps = len(train_dataset) * num_epochs // batch_size
scheduler = get_scheduler(
    "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")

    model.train()
    train_loss = 0

    for batch in tqdm(train_dataloader):
        inputs = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        attention_mask = batch['attention_mask'].to(device)
        outputs = model(inputs, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        train_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
    print("Epoch:", epoch + 1, "Batch Loss:", loss.item())
    print("Train loss:", train_loss / len(train_dataloader))

# Validation
validation_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
model.eval()
val_loss = 0
correct = 0
total = 0
with torch.no_grad():
    for batch in tqdm(validation_dataloader):
            inputs = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(inputs, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            val_loss += loss.item()
            
            prediction = torch.argmax(outputs.logits, dim=1)
            correct += torch.sum(prediction == labels).item()
            total += labels.size(0)
            
val_accuracy = correct / total
print("Validation loss:", val_loss / len(validation_dataloader))
print("Validation accuracy:", val_accuracy)

# Save the fine-tuned model
model.save_pretrained("./bert_cyberhate_model")
tokenizer.save_pretrained("./bert_cyberhate_model")

# Test the model
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
model.eval()
test_loss = 0
correct = 0
total = 0

with torch.no_grad():
    for batch in tqdm(test_dataloader):
        inputs = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        attention_mask = batch['attention_mask'].to(device)
        outputs = model(inputs, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        test_loss += loss.item()
        
        prediction = torch.argmax(outputs.logits, dim=1)
        correct += torch.sum(prediction == labels).item()
        total += labels.size(0)
        
test_accuracy = correct / total
print("Test loss:", test_loss / len(test_dataloader))
print("Test accuracy:", test_accuracy)

# Collect predictions and labels
all_predictions = []
all_labels = []

with torch.no_grad():
    for batch in tqdm(test_dataloader):
        inputs = {key: val.to(device) for key, val in batch.items()}
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=1)
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(inputs['labels'].cpu().numpy())

# Compute metrics
precision = precision_score(all_labels, all_predictions)
recall = recall_score(all_labels, all_predictions)
f1 = f1_score(all_labels, all_predictions)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")

# Test the model
print("Test results:")
def sentiment_score(movie_review):
    token = tokenizer.encode(movie_review, return_tensors = 'pt')
    result = model(token)
    return int(torch.argmax(result.logits))+1
df['label'] = df['text'].apply(lambda x: sentiment_movie_score(x[:512]))
print(df)
print(sentiment_score("I love you"))
print(sentiment_score("😂🖕 you"))
print(sentiment_score("They're all criminals to me 👩🏿👨🏿"))
