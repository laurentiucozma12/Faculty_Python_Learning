import spacy

nlp = spacy.load("en_core_web_sm")

## Example 0
text1="Anna has apples"
doc1=nlp(text1)
from spacy import displacy

## Example 1
# displacy.serve(doc1, style='dep')

## Example 2
# displacy.serve(doc1, style='ent')

## Example 3
# print(displacy.render(doc1, style="ent", page="true"))

## Example 4
print(doc1[0]) # Anna
print([token for token in doc1]) # [Anna, has, apples]
print([token.text for token in doc1]) # ['Anna', 'has', 'apples']

## Example 5
text2="Andrei lives in Cluj-Napoca."
doc2=nlp(text2)
print([token for token in doc2]) # [Andrei, lives, in, Cluj, -, Napoca, .]

## Example 6
text3="Cristina lives in N.Y."
doc3=nlp(text3)
print([token for token in doc3]) # [Cristina, lives, in, N.Y.]

## Example 7
t="Anna has 12 apples and one melon."
d=nlp(t)
print([token.i for token in d])
[0, 1, 2, 3, 4, 5, 6, 7]
print([token.is_alpha for token in d]) # [True, True, False, True, True, True, True, False]
print([token.is_punct for token in d]) # [False, False, False, False, False, False, False, True]
print([token.like_num for token in d]) # [False, False, True, False, False, True, False, False]

## Example 8
text2="Andrei lives in Cluj-Napoca."
nlp.tokenizer.explain(text2) # [('TOKEN', 'Andrei'), ('TOKEN', 'lives'), ('TOKEN', 'in'), ('TOKEN', 'Cluj'), ('INFIX', '-'), ('TOKEN', 'Napoca'), ('SUFFIX', '.')]

## Example 9
text1="Anna has apples"
text2="Andrei lives in Cluj-Napoca."
text3="Cristina lives in N.Y."
text= text1 + ". " + text2 + " " + text3
doc=nlp(text)
print(doc) # Anna has apples. Andrei lives in Cluj-Napoca. Cristina lives in N.Y.

for sent in doc.sents:
    print(sent)
# Anna has apples.
# Andrei lives in Cluj-Napoca.
# Cristina lives in N.Y.

sentences=list(doc.sents)
print(sentences) # [Anna has apples., Andrei lives in Cluj-Napoca., Cristina lives in N.Y.]
span1 = doc[:4]
print(span1) # Anna has apples.
span2 = doc[4:11] # primul index este inclusiv iar al doilea exclusiv
print(span2) # Andrei lives in Cluj-Napoca.
span3 = doc[-4:] # indexul negativ realizează numărarea de la dreapta la stânga
print(span3) # Cristina lives in N.Y.

## Example 9
word=doc1.char_span(0,7)
word #valoarea e None
word=doc1.char_span(0,8)
print(word) # Anna has
print(word[0]) # Anna
word=doc1.char_span(2,7, alignment_mode= "expand")
print(word) # Anna has

## Example 10
print(span2.start) # 4
print(span2.end) # 11
print(span2.start_char) # 17
print(span2.end_char) # 45

## Example 11
token=doc[0] # aici doc e variabila doc în care am salvat textul prelucrat
print(token.doc) # aici doc e proprietate, ce e accesibilă la un obiect de tip token, ce ne returnează documentul din care a fost extras token-ul
# Anna has apples. Andrei lives in Cluj-Napoca. Cristina lives in N.Y.
print(token.sent) # Anna has apples.

## Example 12
for token in d:
    print(token.text, token.pos_)
# Anna PROPN
# has VERB
# 12 NUM
# apples NOUN
# and CCONJ
# one NUM
# melon NOUN
# . PUNCT

## Example 13
print(doc1[1].lemma_)
# have

## Example 14
print(doc.ents) # (Anna, Andrei, Cluj-Napoca, N.Y.)

## Example 15
print(doc1[0].ent_type_) # PERSON

## Example 16
# for ent in doc.ents:
#     print(ent.ent_type_)
# Traceback (most recent call last):
# File "<pyshell#136>", line 2, in <module>
# AttributeError: 'spacy.tokens.span.Span' object has no attribute 'ent_type_'

## Example 17
print([ent.label_ for ent in doc.ents]) # ['PERSON', 'PERSON', 'LOC', 'GPE']

## Example 18
spacy.explain("LOC") # 'Non-GPE locations, mountain ranges, bodies of water'
spacy.explain("GPE") # 'Countries, cities, states'

## Example 19
json_doc=doc1.to_json()
import json
print(json.dumps(json_doc, indent=4))