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
avg train loss: 0.48592027944512667
val loss: 0.3736413549724966, val accuracy: 87.65555555555557
test loss: 0.45815397398391117, test accuracy: 84.24444444444444
----------------------------------------
...
----------------------------------------
Epoch: 7
avg train loss: 0.1560968588325583
val loss: 0.4014184126111407, val accuracy: 89.23333333333333
test loss: 0.44276860112072125, val accuracy: 88.16111111111111
----------------------------------------
Epoch: 8
avg train loss: 0.14510568015102762
val loss: 0.40868822049097314, val accuracy: 89.15
test loss: 0.4309448376139335, val accuracy: 88.82777777777777
----------------------------------------
Epoch: 9
avg train loss: 0.13812587918342614
val loss: 0.4355157840340932, val accuracy: 89.06666666666666
test loss: 0.46008312151325903, val accuracy: 88.57777777777777

----------------------------------------
Epoch: 10
avg train loss: 0.13695546505636835
val loss: 0.4530720585329497, val accuracy: 89.40555555555557
test loss: 0.49575804626343595, val accuracy: 88.74444444444444
----------------------------------------
Epoch: 11
avg train loss: 0.13742327113526698
val loss: 0.46566479140628264, val accuracy: 89.65
test loss: 0.47167895977048224, val accuracy: 89.41111111111111
----------------------------------------
Epoch: 12
avg train loss: 0.13387962260836503
val loss: 0.4197988706693286, val accuracy: 89.65555555555557
test loss: 0.4746199485582959, val accuracy: 88.74444444444444
----------------------------------------
Epoch: 13
avg train loss: 0.12412958373776443
val loss: 0.45670228529294643, val accuracy: 89.82222222222222
test loss: 0.5275128867946478, val accuracy: 88.57777777777777
----------------------------------------
Epoch: 14
avg train loss: 0.12366450859364704
val loss: 0.4480891472221871, val accuracy: 89.90555555555557
test loss: 0.49854352105083916, val accuracy: 88.41111111111111
----------------------------------------
Epoch: 15
avg train loss: 0.11987577051690702
val loss: 0.4446297564179743, val accuracy: 90.07222222222222
test loss: 0.500115554144783, val accuracy: 88.57777777777777
----------------------------------------
Epoch: 16
avg train loss: 0.1188976679590208
val loss: 0.45285178938508275, val accuracy: 90.15
test loss: 0.5024003549455058, val accuracy: 88.41111111111111
----------------------------------------
Epoch: 17
avg train loss: 0.11785403423239282
val loss: 0.47756932590366585, val accuracy: 89.65555555555557
test loss: 0.525707974706238, val accuracy: 88.49444444444444
----------------------------------------
Epoch: 18
avg train loss: 0.11551252618898919
val loss: 0.4675509086661502, val accuracy: 89.9
test loss: 0.5161907441912141, val accuracy: 89.07777777777777
----------------------------------------
Epoch: 19
avg train loss: 0.11399248270408861
val loss: 0.4783930291746704, val accuracy: 89.9
test loss: 0.52922599939416, val accuracy: 88.66111111111111
----------------------------------------
Epoch: 20
avg train loss: 0.1129702885591784
val loss: 0.4780923488369323, val accuracy: 89.9
test loss: 0.5288947856525434, val accura


```

### 6. Evaluation
I used the val-accuracy score to determine the best model, 
then the test-accuracy to check how well it performs within unseen data.

### 7. Final Verdict on Accuracy
The accuracy on the test data was 88.41% (at epoch 16 with 90.15% val accuracy)
which is probably not really suitable to use it yet.

### 8. Outlook
- different transformer architectures could be tested
- hyperparameter optimization (was not enough time within this task)
- find different data sources to wikipedia articles as it is difficult to predict only from the semantics from a term (and ~15% have no matched wikipedia article)