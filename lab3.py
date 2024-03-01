import random
import spacy
nlp = spacy.load("en_core_web_sm")
from spacy.pipeline.textcat import DEFAULT_SINGLE_TEXTCAT_MODEL
# DEFAULT_MULTI_TEXTCAT_MODEL în felul următor
from spacy.pipeline.textcat_multilabel import DEFAULT_MULTI_TEXTCAT_MODEL
config = {
    "threshold": 0.5,
    "model": DEFAULT_SINGLE_TEXTCAT_MODEL
}

textcat = nlp.add_pipe("textcat", config=config)
train_data = [
    ("I love these apples.", {"cats":{"pozitive": 1.0}}),
    ("I recommend these apples. Very tasty.", {"cats": {" pozitive ": 1.0}}),
    ("These are the best apples I ever ate. Will buy again.", {"cats": {" pozitive ": 1.0}}),
    ("Disappointed. These apples have no taste", {"cats": {"negative": 0.0}}),
    ("I didn’t like the smell. Won't buy again", {"cats": {" negative ": 0.0}}),
    ("Very few amount of product for a high price. Don't recommend.", {"cats": {" negative ": 0.0}})
]
from spacy.training import Example
train_examples = [Example.from_dict(nlp.make_doc(text), label) for text,label in train_data]
textcat.add_label("pozitive")
textcat.add_label("negative")

textcat.initialize(lambda: train_examples, nlp=nlp)
epochs=20
with nlp.select_pipes(enable="textcat"):
    optimizer = nlp.resume_training()
for i in range(epochs):
    random.shuffle(train_data)
for text, label in train_data:
    doc = nlp.make_doc(text)
example = Example.from_dict(doc, label)
nlp.update([example], sgd=optimizer)