import spacy
nlp = spacy.load("en_core_web_sm")
from spacy.matcher import Matcher

## Ex 1
t1 = 'Anna is learning Business Process Modelling at Universitatea Babeș Bolyai'
d1 = nlp(t1)
print(d1)

matcher = Matcher(nlp.vocab)
print(matcher)
#se vor regăsi construcții precum Business Process Modelling sau BUSINESS PROCESS MODELLING
pattern = [{'LOWER':'business'},{'LOWER':'process'},{'LOWER':'modelling'}]
matcher.add("BPM_pattern", [pattern])
print(pattern)

matches = matcher(d1) 
for match_id, start, end in matches:
    matched_span = d1[start:end]
print (matched_span.text) # Business Process Modelling
                                               
## Ex 2
text = "The closing session at ISD in Valencia has just announced that the 30th anniversary edition of the International Conference on Information Systems Development will be hosted next year, between August 31 - September 2, by the Faculty of Economics and Business Administration at BabeșBolyai University. The website will be launched early October, the conference will receive submissions until beginning of April."
d = nlp(text)
entitati = d.ents
for i in entitati:
    print(i, i.label_)
# ISD GPE
# Valencia GPE
# 30th ORDINAL
# the International Conference on Information Systems Development ORG
# next year DATE
# between August 31 - September 2 DATE
# the Faculty of Economics and Business Administration ORG
# Babeș-Bolyai University ORG
# early October DATE
# April DATE

## Ex 3
trainset = [
    ("I attended KSEM 2019 Conference",
    {"entities":[(11,31, "CONFERENCE")]}),
    ("I was unable to attend the wedding",
    {"entities":[(23,35, "EVENT")]}),
    ("I attended ISD",
    {"entities":[(11,14, "CONFERENCE")]}),
    ("We welcome submissions for the 20th International Conference on Perspectives in Business Informatics Research (BIR 2021)",
    {"entities":[(27,109,"CONFERENCE"),(111,114,"CONFERENCE")]})
]

ner = nlp.get_pipe('ner')

for text, annotation in trainset:
    for ent in annotation.get("entities"):
        ner.add_label(ent[2])

#întâi adunăm în variabila other_pipes celelalte componente
other_pipes=[pipe for pipe in nlp.pipe_names if pipe !='ner']
#folosim metoda disable_pipes pentru fiecare element din variabila other_pipes
nlp.disable_pipes(*other_pipes)
                  
## Ex 4
#pentru a crea un obiect Example este nevoie să importăm clasa Example din spacy.training
from spacy.training import Example

#este necesar să importăm și pachetul random pentru amestecarea datelor de antrenare
import random

#creăm căteva date de test
trainset = [("I attended KSEM 2019 Conference",{"entities":[(11,31, "CONFERENCE")]}),("I attended ISD",{"entities":[(11,14, "CONFERENCE")]}),("I was unable to attend the wedding",{"entities":[(23,35,"EVENT")]}),("We welcome submissions for the 20th International Conference on Perspectives in Business Informatics Research (BIR 2021)",{"entities":[(23,109,"CONFERENCE"),(111,114,"CONFERENCE")]})]

#dezactivăm celelalte componente din obiectul nlp și pentru componenta NER rămasă începem procesul de antrenare
other_pipes=[pipe for pipe in nlp.pipe_names if pipe !='ner']

with nlp.disable_pipes(*other_pipes):
    # adăugăm noile etichete în componenta NER
    ner = nlp.get_pipe('ner')

for text, annotation in trainset:
    for ent in annotation.get('entities'):
        ner.add_label(ent[2])

#creăm un obiect optimiser
optimizer=nlp.create_optimizer()

#iterăm de 30 de ori prin setul de date, cu cât mai mult cu atât mai bine
for i in range (30):
    random.shuffle(trainset)

for text, annotation in trainset:
    doc = nlp.make_doc(text)

example = Example.from_dict(doc, annotation)
nlp.update([example], sgd=optimizer)

## Ex 5
text="The closing session at ISD in Valencia has just announced that the 30th anniversary edition of the International Conference on Information Systems Development will be hosted next year, between August 31 - September 2, by the Faculty of Economics and Business Administration at BabeșBolyai University. The website will be launched early October, the conference will receive submissions until beginning of April."
d=nlp(text)
entitati=d.ents
for i in entitati:
    print (i, i.label_)
# ISD CONFERENCE
# Valencia GPE
# of the International Conference on Information Systems Development CONFERENCE
# next year DATE
# between August 31 - September 2 DATE
# the CONFERENCE
# Business CONFERENCE
# Babeș CONFERENCE
# early October DATE
# until beginning of April. CONFERENCE
