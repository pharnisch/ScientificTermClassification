import pickle
import pandas as pd
import matplotlib

with open('data/labels.pkl', 'rb') as labels_f, open('data/terms.pkl', 'rb') as terms_f, open('data/texts.pkl', 'rb') as texts_f:
    labels, terms, texts = pickle.load(labels_f), pickle.load(terms_f), pickle.load(texts_f)

# a) First I just wanted to know how many items each file to make sure they have the same amount
print(f"Line counts: labels:{len(labels)}, terms:{len(terms)}, texts:{len(texts)}")

# b) Then I wanted to get a general idea what the data looks like
df = pd.DataFrame(list(zip(labels, terms, texts)), columns=["label", "term", "text"])
print(df)

# c) See the distribution of the labels
print(df.label.value_counts())
# -> So, roughly half of the items are non-scientific,
# the other labels appear equally often, only antibiotic appears less often

# d) Understand data (and possible use-cases) for the additional text
print(f"First texts: {texts[:5]}")
# -> Some labels seem to be unusuable (e.g. 'No Results' or 'DisambiguationError'),
# other data looks more like Wikipedia articles of the term (rather than what I understand as search texts
# - what the task description calls it), with relatively long and (wiki-)formatted text (\n, ==)

# e) I want to examine which texts could be unhelpful and how often they occur
short_texts = [t[0] for t in texts if len(t[0]) <= 20]
print(f"{len(short_texts)/len(texts)} percent of the texts have only 20 characters or less. (thus, are most likely not a Wikipedia article)")
print(set(short_texts))
# -> So, the model won't have the text available at approx. ~15% of the time, because it does not find a Wiki article.
