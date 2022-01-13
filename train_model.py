import pickle
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import AdamW
from transformers import BertTokenizer, BertModel, BertConfig
import regex as re
import random
import numpy as np

with open('data/labels.pkl', 'rb') as labels_f, open('data/terms.pkl', 'rb') as terms_f, open('data/texts.pkl', 'rb') as texts_f:
    labels, terms, texts = pickle.load(labels_f), pickle.load(terms_f), pickle.load(texts_f)

target_dict = {}
for label in labels:
    if label not in target_dict:
        target_dict[label] = len(target_dict)

epochs = 20

# seperate data into train, val, test
ids = list(range(len(labels)))
random.seed(42)
random.shuffle(ids)
split_1 = int(0.8 * len(ids))
split_2 = int(0.9 * len(ids))
train_ids = ids[:split_1]
val_ids = ids[split_1:split_2]
test_ids = ids[split_2:]

# init tokenizer to prepare data for the transformer
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")


# helper function to clean and tokenize
def tokenize_function(sentences, tokenizer):
    cleaned_sentences = [sentence.replace("\n", "").replace("==", "") for sentence in sentences]
    cleaned_sentences = [sentence.replace("No Results", "").replace("DisambiguationError", "") for sentence in cleaned_sentences]
    cleaned_sentences = [re.sub(r'\s+', ' ', sentence).strip() for sentence in cleaned_sentences]
    tokenized = tokenizer(cleaned_sentences, padding="max_length", truncation=True)
    _ids = tokenized["input_ids"]
    _masks = tokenized["attention_mask"]
    return _ids, _masks


# helper function to split data
def get_data_split(remaining_ids):
    _labels = [l for idx, l in enumerate(labels) if idx in remaining_ids]
    _terms = [l for idx, l in enumerate(terms) if idx in remaining_ids]
    _texts = [l for idx, l in enumerate(texts) if idx in remaining_ids]
    _label_ids = [target_dict[l] for l in _labels]
    _inputs = [term[0] + "[SEP]" + text[0] for term, text in zip(_terms, _texts)]
    _inputs, _masks = tokenize_function(_inputs, tokenizer)
    return torch.tensor(_label_ids), torch.tensor(_inputs), torch.tensor(_masks)


train_labels, train_inputs, train_masks = get_data_split(train_ids)
val_labels, val_inputs, val_masks = get_data_split(val_ids)
test_labels, test_inputs, test_masks = get_data_split(test_ids)


class TermClassifier(torch.nn.Module):
    def __init__(self, target_dict):
        super(TermClassifier, self).__init__()
        self.transformer = BertModel.from_pretrained("bert-base-cased")
        self.linear = torch.nn.Linear(768, len(target_dict))  # 768 is hidden_size default of BERT

    def forward(self, input_ids, attention_mask):
        # Feed required information about the term-sep-text-combination into the Transformer
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs[0][:, 0, :]
        # Feed input to classifier to compute logits
        _logits = self.linear(last_hidden_state_cls)
        return _logits

    def predict(self, input_ids, attention_mask):
        _logits = self.forward(input_ids, attention_mask)
        # receive the index that is the most likely one
        index = torch.argmax(_logits, dim=1)
        return index


model = TermClassifier(target_dict)

batch_size = 16

# Create the DataLoader for training set
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# Create the DataLoader for validation set
val_data = TensorDataset(val_inputs, val_masks, val_labels)
val_sampler = SequentialSampler(val_data)
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

# make usage of gpu if available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")
model.to(device)

# Create the optimizer
optimizer = AdamW(model.parameters(),
                  lr=5e-5,    # Default learning rate
                  eps=1e-8    # Default epsilon value
                  )

# Total number of training steps
total_steps = len(train_dataloader) * epochs

# Set up the learning rate scheduler
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)


# evaluation helper method
def evaluate(model, val_dataloader):
    """After the completion of each training epoch, measure the model's performance
    on our validation set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()

    # Tracking variables
    val_accuracy = []
    val_loss = []

    # For each batch in our validation set...
    for batch in val_dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)

        # Compute loss
        loss = loss_fn(logits, b_labels)
        val_loss.append(loss.item())

        # Get the predictions
        preds = torch.argmax(logits, dim=1).flatten()

        # Calculate the accuracy rate
        accuracy = (preds == b_labels).cpu().numpy().mean() * 100
        val_accuracy.append(accuracy)

    # Compute the average accuracy and loss over the validation set.
    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)

    return val_loss, val_accuracy


best_val_accuracy = 0
loss_fn = torch.nn.CrossEntropyLoss()
for epoch_i in range(epochs):
    print(f"Epoch: {epoch_i + 1}")
    total_loss, batch_loss, batch_counts = 0, 0, 0
    model.train()
    for step, batch in enumerate(train_dataloader):
        batch_counts += 1
        b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)
        model.zero_grad()
        logits = model(b_input_ids, b_attn_mask)

        # Compute loss and accumulate the loss values
        loss = loss_fn(logits, b_labels)
        batch_loss += loss.item()
        total_loss += loss.item()

        # Perform a backward pass to calculate gradients
        loss.backward()

        # Update parameters and the learning rate
        optimizer.step()
        scheduler.step()

    # Calculate the average loss over the entire training data
    avg_train_loss = total_loss / len(train_dataloader)
    print(f"avg train loss: {avg_train_loss}")

    # After the completion of each training epoch, measure the model's performance
    # on our validation set.
    val_loss, val_accuracy = evaluate(model, val_dataloader)
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), f"BERT_classifier_best_val_accuracy")
    print(f"val loss: {val_loss}, val accuracy: {val_accuracy}")
    test_loss, test_accuracy = evaluate(model, test_dataloader)
    print(f"test loss: {test_loss}, test accuracy: {test_accuracy}")
    print("----------------------------------------")

