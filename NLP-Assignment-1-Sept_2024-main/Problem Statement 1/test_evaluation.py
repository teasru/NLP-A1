import torch
from transformers import BertTokenizerFast, BertForTokenClassification
import os

# Defining labels and constants
LABEL_LIST = ['ARG1', 'ARG2', 'NONE', 'REL', 'TIME', 'LOC']

# Load the tokenizer and model
model_path = "saved_model"  # Update as necessary
tokenizer = BertTokenizerFast.from_pretrained(model_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model
model = BertForTokenClassification.from_pretrained(model_path)
model.to(device)

# Generate predictions and tuple grouping
def generate_predictions(sentences, model, tokenizer, device):
    model.eval()
    all_extractions = []

    with torch.no_grad():
        for sentence in sentences:
            tokenized_input = tokenizer(sentence, truncation=True, padding='max_length', max_length=512, return_tensors="pt")
            input_ids = tokenized_input['input_ids'].to(device)
            attention_mask = tokenized_input['attention_mask'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)
            pred = predictions[0].cpu().numpy()
            tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())

            # Grouping tuples
            extractions = []
            current_tuple = {"subject": None, "relation": None, "object": None, "time": None, "location": None}
            
            for token, label in zip(tokens, pred):
                label_name = LABEL_LIST[label]
                
                if label_name == 'ARG1':
                    current_tuple["subject"] = token
                elif label_name == 'ARG2':
                    current_tuple["object"] = token
                elif label_name == 'REL':
                    current_tuple["relation"] = token
                elif label_name == 'TIME':
                    current_tuple["time"] = token
                elif label_name == 'LOC':
                    current_tuple["location"] = token
                
                # If the current tuple is complete (has subject, relation, object)
                if current_tuple["subject"] and current_tuple["relation"] and current_tuple["object"]:
                    extractions.append(current_tuple.copy())
                    current_tuple = {"subject": None, "relation": None, "object": None, "time": None, "location": None}

            all_extractions.append((sentence, extractions))

    return all_extractions

# Save extractions to a file in the required format
def save_extractions_to_file(extractions, filename):
    with open(filename, 'w') as f:
        for sentence, tuples in extractions:
            for extraction in tuples:
                rel = extraction.get('relation', '')
                arg1 = extraction.get('subject', '')
                arg2 = extraction.get('object', '')
                loc = extraction.get('location', '')
                time = extraction.get('time', '')

                # Writing in tab-separated format with fixed probability of 1.00
                f.write(f"{sentence}\t1.00\t{rel}\t{arg1}\t{arg2}\t{time}\t{loc}\n")

# Load test sentences from the test file
def load_test_sentences(file_path):
    sentences = []
    try:
        with open(file_path, "r", encoding='utf-8') as file:
            for line in file:
                sentence = line.strip()
                if sentence:
                    sentences.append(sentence)
    except Exception as e:
        print(f"Error reading test file: {e}")
    return sentences

# Run extractions on test data and save
test_file = "test_data.txt"  # Change the file path/name if necessary
test_sentences = load_test_sentences(test_file)
extractions = generate_predictions(test_sentences, model, tokenizer, device)
save_extractions_to_file(extractions, 'your_output.txt')

# Evaluate using CaRB
os.system("python carb.py --gold=data/gold/test.tsv --out=output.txt --tabbed=your_output.txt")

