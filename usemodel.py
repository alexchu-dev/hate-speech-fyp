from transformers import BertForSequenceClassification, BertTokenizer
import torch

# Load the trained model and tokenizer
model = BertForSequenceClassification.from_pretrained("./bert_cyberhate_model")
tokenizer = BertTokenizer.from_pretrained("./bert_cyberhate_model")

# Set the device (MPS, CUDA, or CPU)
device = torch.device("mps") if torch.backends.mps.is_available() else (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)
model.to(device)

# Input text
text = "fuck these black gay"

# Tokenize and prepare for model input
inputs = tokenizer(
    text,
    padding="max_length",  # Ensure consistent input size
    truncation=True,       # Truncate if input is too long
    max_length=128,        # Match the max length used during training
    return_tensors="pt"    # Return PyTorch tensors
)

# Move the tokenized input to the same device as the model
inputs = {key: val.to(device) for key, val in inputs.items()}

# Set the model to evaluation mode
model.eval()

# Perform inference
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits  # Output logits (raw scores)
    print(logits)

# Get the predicted class
predicted_class = torch.argmax(logits, dim=-1).item()
print(f"Predicted class: {predicted_class}")

label_map = {0: "Negative", 1: "Positive"}  # Example label map
print(f"Predicted label: {label_map[predicted_class]}")
