import torch
from transformers import BertTokenizerFast, BertForTokenClassification, AdamW
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import os

#labels and constants defining
LABEL_LIST = ['ARG1', 'ARG2', 'NONE', 'REL', 'TIME', 'LOC']
label_map = {label: idx for idx, label in enumerate(LABEL_LIST)}

#bert tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#loading data from the file
def load_data_from_file(file_path):
    sentences, label_sets = [], []
    try:
        with open(file_path, "r", encoding='utf-8') as file:
            current_sentence, current_label_sets = None, []
            for line in file:
                line = line.strip()
                if not line:
                    continue
                first_word = line.split()[0]
                if first_word in LABEL_LIST:
                    current_label_sets.append(line.split())
                else:
                    if current_sentence:
                        sentences.append(current_sentence)
                        label_sets.append(current_label_sets)
                        current_label_sets = []
                    current_sentence = line.split()
            if current_sentence:
                sentences.append(current_sentence)
                label_sets.append(current_label_sets)
    except Exception as e:
        print(f"Error reading file: {e}")
    return sentences, label_sets

#flattening the dataset
def flatten_dataset(sentences, label_sets):
    flat_sentences, flat_labels = [], []
    for sentence, labels in zip(sentences, label_sets):
        for label_set in labels:
            flat_sentences.append(sentence)
            flat_labels.append(label_set)
    return flat_sentences, flat_labels

#tokenize and preserve labels
def tokenize_and_preserve_labels(sentence, labels, tokenizer):
    tokenized_inputs = tokenizer(sentence, truncation=True, padding='max_length', is_split_into_words=True, max_length=512)
    word_ids = tokenized_inputs.word_ids()
    aligned_labels = []
    previous_word_idx = None

    for word_idx in word_ids:
        if word_idx is None:
            aligned_labels.append(-100)  # Padding token
        elif word_idx != previous_word_idx:
            if word_idx < len(labels):
                aligned_labels.append(label_map[labels[word_idx]])
            else:
                aligned_labels.append(-100)  # No corresponding label
        else:
            aligned_labels.append(-100)  # If it's the same word, ignore it
        previous_word_idx = word_idx

    tokenized_inputs['labels'] = aligned_labels
    return tokenized_inputs

#dataset tokenizing
def tokenize_dataset(sentences, labels, tokenizer):
    tokenized_data = []
    for sentence, label in zip(sentences, labels):
        tokenized_inputs = tokenize_and_preserve_labels(sentence, label, tokenizer)
        tokenized_data.append(tokenized_inputs)
    return tokenized_data

#custom dataset class
class CustomDataset(Dataset):
    def __init__(self, tokenized_inputs):
        self.inputs = tokenized_inputs

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {key: torch.tensor(val) for key, val in self.inputs[idx].items()}

#collate function
def collate_fn(batch):
    input_ids = pad_sequence([item['input_ids'] for item in batch], batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = pad_sequence([item['attention_mask'] for item in batch], batch_first=True, padding_value=0)
    labels = pad_sequence([item['labels'] for item in batch], batch_first=True, padding_value=-100)
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

#load and tokenize data
file_path = "original_cleaned3.txt"  
sentences, label_sets = load_data_from_file(file_path)
flat_sentences, flat_labels = flatten_dataset(sentences, label_sets)

#tokenize the dataset
train_data = tokenize_dataset(flat_sentences, flat_labels, tokenizer)

#load dataset
train_dataset = CustomDataset(train_data)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

#initialize model
model = BertForTokenClassification.from_pretrained('bert-base-cased', num_labels=len(LABEL_LIST))
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)

#training loop
def train_model(model, train_loader, optimizer, epochs=3):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")

#save the model
def save_model(model, tokenizer, model_path):
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

#train the model
train_model(model, train_loader, optimizer, epochs=3)

#save the model for later use
save_model(model, tokenizer, "saved_model")
