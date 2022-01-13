# Scientific Named-Entity Recognition (LabTwin Application Task)

## Setup

```
pip install -r requirements.txt
```

### Usage

data exploration:
```
python explore_data.py
```

training:
```
python train_model.py
```

## Information about the Implementation Process

### 1. Exploration
- roughly half of the items are non-scientific, the other labels appear equally often, only antibiotic appears less often
- ~15% of items have no helpful text

(see comments in explore_data.py)

### 2. Data Preparation
Remove character sequences that do not help the model:
- Remove Wikimarkup: "\n" and "=="
- Remove falsy values from texts: "No text", "DisambiguationError"
- Strip blank spaces

Tokenization for Transformer (BERT): tokenize, adding tokens, padding, truncating

(see train_model > tokenize_function)

### 3. Model Choice
- transformer architecture as it is SOTA for NLP.
    - it also enables to feed two parts of input with [SEP],
    attention mechanism learns to employ knowledge from both input parts
- BERT as an easy first implementation 
- linear layer on top of [CLS]-token logits for the classification, using this token index as a central target where the transformer will be fine-tuned to propagate the required information to

(see train_model > TermClassifier)
### 4. Feature Engineering
- term is the main feature and will be fed as the first part to the transformer
- text could be empty, but important for the task as the term semantic alone would tell very little about its scientific domain; it is fed as a second optional part to the transformer (can be an empty string)

### 5. Training
- data split: train, val, test; to make sure that one gets a realistic accuracy score for unseen data
- 20 epochs
- optimizer: AdamW, 5e-5 lr
- scheduler: linear with warmup
- plotting loss and accuracy to see progress/choose best model

On GPU Tesla V100-PCIE-32GB:
```
Epoch: 1
avg train loss: 0.45981357934108624
val loss: 0.38452588010268907, val accuracy: 86.5
test loss: 0.38804003052568686, test accuracy: 87.16666666666667
----------------------------------------
Epoch: 2
avg train loss: 0.32162566339674714
val loss: 0.4077319192502182, val accuracy: 87.66666666666667
test loss: 0.4174609709744497, test accuracy: 87.5
----------------------------------------
Epoch: 3
avg train loss: 0.2521410969590458
val loss: 0.33876629498379773, val accuracy: 88.58333333333333
test loss: 0.3609540869119034, test accuracy: 88.75
----------------------------------------
Epoch: 4
avg train loss: 0.20726745444039504
val loss: 0.3520806852184857, val accuracy: 89.32777777777777
test loss: 0.4039795075293902, test accuracy: 88.49444444444444
----------------------------------------
Epoch: 5
avg train loss: 0.18417148951635076
val loss: 0.36441500576833885, val accuracy: 88.07777777777777
test loss: 0.393384957042775, test accuracy: 88.33333333333333
----------------------------------------
Epoch: 6
avg train loss: 0.16998945107246982
val loss: 0.36415901160575836, val accuracy: 89.16666666666667
test loss: 0.3725149495061487, test accuracy: 89.41666666666667
----------------------------------------
Epoch: 7
avg train loss: 0.15714695367447953
val loss: 0.3906371349274074, val accuracy: 89.16666666666667
test loss: 0.3880362138935258, test accuracy: 90.25
----------------------------------------
Epoch: 8
avg train loss: 0.1507952446648172
val loss: 0.41255322317408477, val accuracy: 89.16666666666667
test loss: 0.38903524216618585, test accuracy: 89.08333333333333
----------------------------------------
Epoch: 9
avg train loss: 0.14529295997015046
val loss: 0.4172161076998115, val accuracy: 89.57777777777777
test loss: 0.40865574848908487, test accuracy: 90.08333333333333
----------------------------------------
Epoch: 10
avg train loss: 0.13440908850105188
val loss: 0.4193162186234743, val accuracy: 89.07777777777777
test loss: 0.43224049615319626, test accuracy: 89.41666666666667
----------------------------------------

```

### 6. Evaluation
I used the val-accuracy score to determine the best model, 
then the test-accuracy to check how well it performs within unseen data.

### 7. Final Verdict on Accuracy
The accuracy on the test data was 90.08% (at epoch 9 with 89.58% val accuracy)
which is probably not really suitable to use it yet.

### 8. Outlook
- different transformer architectures could be tested
- hyperparameter optimization (was not enough time within this task)
- find different data sources to wikipedia articles as it is difficult to predict only from the semantics from a term (and ~15% have no matched wikipedia article)