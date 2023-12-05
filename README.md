
# webanno_tsv

A python library to parse TSV files as produced by the [webanno Software](https://github.com/webanno/webanno) and as described [in their documentation](https://zoidberg.ukp.informatik.tu-darmstadt.de/jenkins/job/WebAnno%20%28GitHub%29%20%28master%29/de.tudarmstadt.ukp.clarin.webanno$webanno-webapp/doclinks/1/#sect_webannotsv).

The following features are supported:

* WebAnno's UTF-16 indices for Text indices
* Webanno's [escape sequences](https://zoidberg.ukp.informatik.tu-darmstadt.de/jenkins/job/WebAnno%20%28GitHub%29%20%28master%29/de.tudarmstadt.ukp.clarin.webanno$webanno-webapp/doclinks/1/#_reserved_characters)
* Multiple span annotation layers with multiple fields
* Span annotations over multiple tokens and sentences
* Multiple Annotations per field (stacked annotations)
* Disambiguation IDs (here called `label_id`)
* Relations

The following is __not supported__:

* Chain annotations
* Sub-Token annotations (ignored on reading)


## Installation

```sh
pip install git+https://github.com/ArneBinder/webanno_tsv
```

## Examples

To construct a Document with annotations you could do:

```python
from webanno_tsv import Document, Layer, SpanLayerDefinition, RelationLayerDefinition, Sentence, Token, SpanAnnotation, RelationAnnotation

# create a sentence with two tokens
sent1 = Sentence(text="First sentence").add_token(Token(start=0, end=5), expected_text="First").add_token(Token(start=6, end=14), expected_text="sentence")
sent2 = Sentence(text="Second sentence").add_token(Token(start=15, end=21), expected_text="Second").add_token(Token(start=22, end=30), expected_text="sentence")

# create a new document with three layers:
layers = [
    Layer(SpanLayerDefinition(name='Layer1', features=('Field1',))),
    Layer(SpanLayerDefinition(name='Layer2', features=('Field2', 'Field3'))),
    Layer(RelationLayerDefinition(name='Layer3', features=('Field4', ), base='Layer2')),
]
doc = Document(layers)

# add the sentences to the document
doc = doc.add_sentence(sent1).add_sentence(sent2)

print(doc.text)
# Prints:
# First sentence
# Second sentence

# add span annotations to the document
# to Layer1
span1 = SpanAnnotation(values=('ABC',), tokens=doc.tokens[1:2])
doc = doc.add_annotation("Layer1", span1)
span2 = SpanAnnotation(values=('DEF',), tokens=doc.tokens[3:4])
doc = doc.add_annotation("Layer1", span2)
# to Layer2
span3 = SpanAnnotation(values=(None, 'UVW'), tokens=doc.tokens[0:1])
doc = doc.add_annotation("Layer2", span3)
span4 = SpanAnnotation(values=(None, 'XYZ'), tokens=doc.tokens[1:3])
doc = doc.add_annotation("Layer2", span4)

# add a relation annotation to the document
rel = RelationAnnotation(values=('R',), source=span3, target=span4)
doc = doc.add_annotation("Layer3", rel)
```

The call to `doc.tsv()` then returns a string:

```
#FORMAT=WebAnno TSV 3.3
#T_SP=Layer1|Field1
#T_SP=Layer2|Field2|Field3
#T_RL=Layer3|Field4|BT_Layer2


#Text=First sentence
1-1	0-5	First	_	*	UVW	_	_
1-2	6-14	sentence	ABC	*[1]	XYZ[1]	_	_

#Text=Second sentence
2-1	15-21	Second	_	*[1]	XYZ[1]	R	1-1[0_1]
2-2	22-30	sentence	DEF	_	_	_	_

```

Supposing that you have a file with the output above as input you could do:

```python
from webanno_tsv import Document

doc = Document.from_file('/tmp/input.tsv')

print(doc.text)
# Prints:
# First sentence
# Second sentence

for sent in doc.sentences:
    print(sent)

# Prints:
# Sentence(text='First sentence', tokens=(Token(start=0, end=5), Token(start=6, end=14)))
# Sentence(text='Second sentence', tokens=(Token(start=15, end=21), Token(start=22, end=30)))

for tok in doc.tokens:
    print(tok)
    
# Prints:
# Token(start=0, end=5)
# Token(start=6, end=14)
# Token(start=15, end=21)
# Token(start=22, end=30)

for annotation in doc.layers['Layer2']:
    print(annotation)

# Prints:
# Layer2 Field3 XYZ

print(set(doc.layers))

# Prints:
# {'Layer1', 'Layer2', 'Layer3'}

for ann in doc.layers["Layer2"]:
    print(ann)
    
# Prints:
# SpanAnnotation(values=(None, 'UVW'), tokens=(Token(start=0, end=5),))
# SpanAnnotation(values=(None, 'XYZ'), tokens=(Token(start=6, end=14), Token(start=15, end=21)))

for ann in doc.layers["Layer3"]:
    print(ann)

# Prints:
# RelationAnnotation(values=('R',), source=SpanAnnotation(values=(None, 'UVW'), tokens=(Token(start=0, end=5),)), target=SpanAnnotation(values=(None, 'XYZ'), tokens=(Token(start=6, end=14), Token(start=15, end=21))))
```

__Possible Gotcha__: The classes in this library are read-only dataclasses ([dataclasses with `frozen=True`](https://docs.python.org/3/library/dataclasses.html#dataclasses.dataclass)).

This means that their fields are not settable. You can create new versions however with [`dataclasses.replace()`](https://docs.python.org/3/library/dataclasses.html#dataclasses.replace).

```py
from dataclasses import replace

t1 = Token(start=0, end=4)
t2 = replace(t1, start=1)
```


## Development

Run the tests with:

```sh
python -m unittest test/*.py
```

PRs always welcome!
