import pickle
import torch
from tqdm.auto import tqdm
from transformers import AdamW
from transformers import BertTokenizer, BertModel, BertConfig
import regex as re

with open('data/labels.pkl', 'rb') as labels_f, open('data/terms.pkl', 'rb') as terms_f, open('data/texts.pkl', 'rb') as texts_f:
    labels, terms, texts = pickle.load(labels_f), pickle.load(terms_f), pickle.load(texts_f)

epochs = 2
max_items = 10
labels = labels[:max_items]
terms = terms[:max_items]
texts = texts[:max_items]

def tokenize_function(sentences, tokenizer):
    cleaned_sentences = [sentence.replace("\n", "").replace("==", "") for sentence in sentences]
    cleaned_sentences = [sentence.replace("No Results", "").replace("DisambiguationError", "") for sentence in cleaned_sentences]
    cleaned_sentences = [re.sub(r'\s+', ' ', sentence).strip() for sentence in cleaned_sentences]
    tokenized = tokenizer(cleaned_sentences, padding="max_length", truncation=True)
    ids = tokenized["input_ids"]
    masks = tokenized["attention_mask"]
    return ids, masks


target_dict = {}
for label in labels:
    if label not in target_dict:
        target_dict[label] = len(target_dict)
label_ids = [target_dict[l] for l in labels]

inputs = [term[0] + "[SEP]" + text[0] for term, text in zip(terms, texts)]

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
train_inputs, train_masks = tokenize_function(inputs, tokenizer)


class TermClassifier(torch.nn.Module):
    def __init__(self, target_dict):
        super(TermClassifier, self).__init__()
        #self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
        self.transformer = BertModel.from_pretrained("bert-base-cased")
        self.linear = torch.nn.Linear(768, len(target_dict))  # hidden_size default of BERT

    def forward(self, input_ids, attention_mask):
        # Feed input to BERT
        #tokenized = self.tokenize_function(batch)
        #input_ids = torch.tensor(tokenized["input_ids"])
        #attention_mask = torch.tensor(tokenized["attention_mask"])
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs[0][:, 0, :]
        # Feed input to classifier to compute logits
        logits = self.linear(last_hidden_state_cls)
        return logits

    def predict(self, input_ids, attention_mask):
        logits = self.forward(input_ids, attention_mask)
        print(logits)
        index = torch.argmax(logits, dim=1)
        print(index)


model = TermClassifier(target_dict)

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup

# Convert other data types to torch.Tensor
train_labels = torch.tensor(label_ids)
#val_labels = torch.tensor(y_val)
train_inputs = torch.tensor(train_inputs)
train_masks = torch.tensor(train_masks)

batch_size = 16

# Create the DataLoader for training set
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# Create the DataLoader for validation set
#val_data = TensorDataset(val_inputs, val_masks, val_labels)
#val_sampler = SequentialSampler(val_data)
#val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)


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
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0, # Default value
                                            num_training_steps=total_steps)

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

        # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update parameters and the learning rate
        optimizer.step()
        scheduler.step()

    # Calculate the average loss over the entire training data
    avg_train_loss = total_loss / len(train_dataloader)
    print(f"avg train loss: {avg_train_loss}")