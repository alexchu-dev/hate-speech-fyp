import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split

file_path = "datasets\HatemojiBuild-train.csv"
df = pd.read_csv(file_path)
df_test = pd.read_csv("datasets\HatemojiBuild-test.csv")

# print(df.head())

train_text, val_text, train_label, val_label = train_test_split(df['text'], df['label_gold'], test_size=0.2, random_state=42)
test_text, test_label = df_test['text'], df_test['label_gold']

print("Train size:", len(train_text))
print("Val size:", len(val_text))
print("Test size:", len(test_text))

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
def tokenize_text(text):
    return tokenizer(list(text), padding='max_length', truncation=True, max_length=128, return_tensors="pt")

train_encodings = tokenize_text(train_text)
val_encodings = tokenize_text(val_text)
test_encodings = tokenize_text(test_text)



train_label = torch.tensor(train_label.values)
val_label = torch.tensor(val_label.values)
test_label = torch.tensor(test_label.values)

print(train_label)
print(len(test_label))

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
    
train_dataset = EmojiDataset(train_encodings, train_label)
val_dataset = EmojiDataset(val_encodings, val_label)
test_dataset = EmojiDataset(test_encodings, val_label)

# Model architecture
from transformers import BertForSequenceClassification
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)


from transformers import AdamW
from transformers import get_scheduler
optimizer = AdamW(model.parameters(), lr=5e-5)


# Train the model
from torch.utils.data import DataLoader
batch_size = 16
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

num_epochs = 4

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
        optimizer.zero_grad()
        attention_mask = batch['attention_mask'].to(device)
        outputs = model(inputs, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        loss.backward()
        optimizer.step()
    print("Epoch:", epoch, "Loss:", loss.item())

model.eval()
for batch in validation_dataloader:
    with torch.no_grad():
        inputs = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(inputs, labels=labels)
        val_loss = outputs.loss