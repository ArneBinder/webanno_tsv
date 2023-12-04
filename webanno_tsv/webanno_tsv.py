import abc
import csv
import itertools
import re
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Any

NO_LABEL_ID = -1

COMMENT_RE = re.compile('^#')
SPAN_LAYER_DEF_RE = re.compile(r'^#T_SP=([^|]+)\|(.*)$')
RELATION_LAYER_DEF_RE = re.compile(r'^#T_RL=([^|]+)\|(.*)$')
RELATION_BASE_LAYER = re.compile(r'^BT_(.+)$')
SENTENCE_RE = re.compile('^#Text=(.*)')
FIELD_EMPTY_RE = re.compile('^[_*]')
#FIELD_WITH_ID_RE = re.compile(r'(.*)\[([0-9]+)(?:_([0-9]+))?]$')
FIELD_WITH_ID_RE = re.compile(r'(.*)\[([0-9]*)]$')
SUB_TOKEN_RE = re.compile(r'[0-9]+-[0-9]+\.[0-9]+')
RELATION_SOURCE_RE = re.compile(r'^([0-9]+-[0-9]+)(?:\[([0-9]+)_([0-9]+)\])?$')

HEADERS = ['#FORMAT=WebAnno TSV 3.3']

TOKEN_FIELDNAMES = ['sent_tok_idx', 'offsets', 'token']

# Strings that need to be escaped with a single backslash according to Webanno Appendix B
RESERVED_STRS = ['\\', '[', ']', '|', '_', '->', ';', '\t', '\n', '*']

# Multiline sentences are split on this character per Webanno Appendix B
MULTILINE_SPLIT_CHAR = '\f'

# character used to pad text between sentences
SENTENCE_PADDING_CHAR = '\n'


def dict_of_lists_to_list_of_dicts(d: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """
    Convert a dictionary of lists to a list of dictionaries, where each
    dictionary in the list has the same keys as the original dictionary,
    but the values are the elements of the lists at the same index.
    """
    return [dict(zip(d.keys(), t)) for t in zip(*d.values())]


class WebannoTsvDialect(csv.Dialect):
    delimiter = '\t'
    quotechar = None  # disables escaping
    doublequote = False
    skipinitialspace = False
    lineterminator = '\n'
    quoting = csv.QUOTE_NONE


@dataclass(frozen=True, order=True)
class Token:
    sentence_idx: int
    idx: int
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class Sentence:
    idx: int
    text: str
    tokens: Tuple[Token, ...]


@dataclass(frozen=True, order=True)
class AnnotationPart:
    tokens: Tuple[Token, ...]
    layer: 'LayerDefinition'
    field: str
    label: str
    label_id: int = NO_LABEL_ID

    @property
    def start(self):
        return self.tokens[0].start

    @property
    def end(self):
        return self.tokens[-1].end

    @property
    def text(self):
        return ' '.join([t.text for t in self.tokens])

    @property
    def token_texts(self):
        return [token.text for token in self.tokens]

    @property
    def has_label_id(self):
        return self.label_id != NO_LABEL_ID

    def should_merge(self, other: 'AnnotationPart') -> bool:
        return self.has_label_id and other.has_label_id \
               and self.label_id == other.label_id \
               and self.label == other.label \
               and self.field == other.field \
               and self.layer == other.layer

    def merge(self, *other: 'AnnotationPart') -> 'AnnotationPart':
        return replace(self, tokens=tuple(token_sort(list(self.tokens) + [t for o in other for t in o.tokens])))

    @property
    def annotation_id(self) -> Optional[str]:
        if self.label_id == NO_LABEL_ID:
            if len(self.tokens) != 1:
                raise ValueError(f"Cannot create annotation id for multi-token annotation without label id: {self}")
            else:
                return f"{self.tokens[0].sentence_idx}-{self.tokens[0].idx}"
        else:
            return f"{self.label_id}"


@dataclass(frozen=True)
class Annotation(abc.ABC):
    id: str
    features: Dict[str, Any]


@dataclass(frozen=True)
class SpanAnnotation(Annotation):
    tokens: Tuple[Token, ...]


@dataclass(frozen=True)
class RelationAnnotation(Annotation):
    source: SpanAnnotation
    target: SpanAnnotation


@dataclass(frozen=True, order=True)
class LayerDefinition(abc.ABC):
    name: str
    fields: Tuple[str, ...]

    @abc.abstractmethod
    def as_header(self) -> str:
        pass

    def as_columns(self) -> List[str]:
        return [f'{self.name}|{field}' for field in self.fields]

    @staticmethod
    def from_lines(lines: List[str]) -> List['LayerDefinition']:
        return SpanLayer.from_lines(lines) + RelationLayer.from_lines(lines)

    def read_annotations(self, token: Token, row: Dict) -> List['AnnotationPart']:
        fields_values = [(field, val) for field in self.fields for val in _read_annotation_field(row, self, field)]
        fields_labels_ids = [(f, _read_label_and_id(val)) for f, val in fields_values]
        fields_labels_ids_filtered = [(f, label, lid) for (f, (label, lid)) in fields_labels_ids if label != '']

        return [AnnotationPart(tokens=(token,), layer=self, field=field, label=label, label_id=lid) for
                field, label, lid in fields_labels_ids_filtered]

    @abc.abstractmethod
    def new_annotation(
        self, id: str, previous_annotations: Dict['LayerDefinition', List[Annotation]], tokens: List[Token],
        **features
    ) -> 'Annotation':
        pass

    @abc.abstractmethod
    def annotation_to_parts(self, annotation: Annotation) -> List[AnnotationPart]:
        pass

    def annotations_to_parts(self, annotations: Sequence[Annotation]) -> List[AnnotationPart]:
        result = []
        for annotation in annotations:
            result += self.annotation_to_parts(annotation)
        return result


@dataclass(frozen=True)
class SpanLayer(LayerDefinition):

    @staticmethod
    def from_lines(lines: List[str]) -> List['SpanLayer']:
        span_matches = [SPAN_LAYER_DEF_RE.match(line) for line in lines]
        layers = [SpanLayer(name=m.group(1), fields=tuple(m.group(2).split('|'))) for m in span_matches if m]
        return layers

    def as_header(self) -> str:
        """
            Example:
                ('one', ['x', 'y', 'z']) => '#T_SP=one|x|y|z'
            """
        name = self.name + '|' + '|'.join(self.fields)
        return f'#T_SP={name}'

    def new_annotation(
        self, id: str, previous_annotations: Dict['LayerDefinition', List[Annotation]], tokens: Tuple[Token, ...],
        **features
    ) -> 'Annotation':
        return SpanAnnotation(id=id, tokens=tokens, features=features)

    def annotation_to_parts(self, annotation: SpanAnnotation) -> List[AnnotationPart]:
        label_id = int(annotation.id if "-" not in annotation.id else NO_LABEL_ID)
        return [
            AnnotationPart(
                tokens=annotation.tokens, layer=self, field=field, label=annotation.features[field], label_id=label_id
            )
            for field in self.fields
            if field in annotation.features
        ]


@dataclass(frozen=True)
class RelationLayer(LayerDefinition):

    @staticmethod
    def from_lines(lines: List[str]) -> List['RelationLayer']:
        relation_matches = [RELATION_LAYER_DEF_RE.match(line) for line in lines]
        layers = [RelationLayer(name=m.group(1), fields=tuple(m.group(2).split('|'))) for m in relation_matches if m]
        return layers

    def as_header(self) -> str:
        """
            Example:
                ('one', ['x', 'y', 'z']) => '#T_RL=one|x|y|z'
            """
        name = self.name + '|' + '|'.join(self.fields)
        return f'#T_RL={name}'

    @property
    def source_field(self) -> str:
        return self.fields[-1]

    @property
    def base_name(self) -> str:
        match = RELATION_BASE_LAYER.match(self.source_field)
        return match.group(1)

    @property
    def value_fields(self) -> Tuple[str]:
        return self.fields[:-1]

    def new_annotation(
        self, id: str, previous_annotations: Dict['LayerDefinition', List[Annotation]], tokens: List[Token],
        **features
    ) -> 'Annotation':

        previous_layers_by_id = {layer.name: layer for layer in previous_annotations.keys()}
        base_annotations_by_id = {a.id: a for a in previous_annotations[previous_layers_by_id[self.base_name]]}

        # Assume any id and "1-17[1_0]" as example match. This has source token-sentence id (1-17) and (1) and
        # (0) as target and source disambiguation ids, respectively. The "0" source disambiguation id indicates,
        # that no disambiguation is used, we will use the source token-sentence id in this case. For the target,
        # we will use the respective disambiguation id (1). Thus, the final annotation will be: (1-17) -> (1).
        #
        # Assume "18-2" as id and "18-3" as example match. The match has source token id (18-3) and implicit target
        # (0) and source (0) disambiguation ids. We handle the source id as above and use the id of this annotation
        # as target id because this is the token-sentence id where the annotation is located. Thus, the final
        # annotation will be: (18-3) -> (18-2).

        source_match = RELATION_SOURCE_RE.match(features[self.source_field])

        source_id = source_match.group(2) or "0"
        target_id = source_match.group(3) or "0"
        if source_id == "0":
            source_id = source_match.group(1)
        if target_id == "0":
            target_id = id

        source = base_annotations_by_id[source_id]
        target = base_annotations_by_id[target_id]

        feature_values = {field: features[field] for field in self.value_fields}
        return RelationAnnotation(id=id, source=source, target=target, features=feature_values)

    def annotation_to_parts(self, annotation: RelationAnnotation) -> List[AnnotationPart]:
        # annotate only the first token
        tokens = annotation.target.tokens[:1]
        label_id = int(annotation.id if "-" not in annotation.id else NO_LABEL_ID)
        result = [
            AnnotationPart(
                tokens=tokens,
                layer=self,
                field=field,
                label=annotation.features[field],
                label_id=label_id,
            )
            for field in self.value_fields
            if field in annotation.features
        ]
        source_tokens = annotation.source.tokens[0]
        source = f"{source_tokens.sentence_idx}-{source_tokens.idx}"
        source_id = annotation.source.id if "-" not in annotation.source.id else "0"
        target_id = annotation.target.id if "-" not in annotation.target.id else "0"
        if not (source_id == "0" and target_id == "0"):
            source += f"[{source_id}_{target_id}]"

        source_annotation_part = AnnotationPart(
            tokens=tokens, layer=self, field=self.source_field, label=source, label_id=label_id
        )
        result.append(source_annotation_part)
        return result


@dataclass(frozen=True, eq=False)
class Document:
    """
    Document binds together text features (Token, Sentence) with Annotations
    over the text. layer definitions is a tuple of layer and field names that
    defines the annotation.layer and annotation.field names when reading tsv.
    When writing, the layer definitions define which annotations are written
    and in what order.

    Example:
    Given a tsv file with lines like these:

        1-9	36-43	unhappy	JJ	abstract	negative

    You could invoke Document() with layers=
         [
            SpanLayer('l1', ('POS',)),
            SpanLayer('l2', ('category', 'opinion')),
         ]


    allowing you to retrieve the annotation for 'abstract' within:

        doc.match_annotations(layer='l2', field='category')

    If you want to suppress output of the 'l2' layer when writing the
    document you could do:

        doc = dataclasses.replace(doc, layers=[SpanLayer('l1', ('POS',))])
        doc.tsv()
    """
    layers: Sequence[LayerDefinition]
    sentences: Sequence[Sentence]
    tokens: Sequence[Token]
    annotation_parts: Sequence[AnnotationPart]
    annotations: Dict[LayerDefinition, Sequence[Annotation]]
    path: str = ''

    def __post_init__(self):
        object.__setattr__(self, '_layers_by_name', {layer.name: layer for layer in self.layers})

    @property
    def text(self) -> str:
        # we need to use the sentence offsets to reconstruct padding in between them
        result = ''
        for sentence in self.sentences:
            start = sentence.tokens[0].start
            result += SENTENCE_PADDING_CHAR * (start - len(result))
            result += sentence.text
        return result

    @property
    def new_annotation_parts(self) -> Sequence[AnnotationPart]:
        annotation_parts = []
        for layer, annotations in self.annotations.items():
            annotation_parts += layer.annotations_to_parts(annotations)
        return annotation_parts

    @classmethod
    def empty(cls, layers: Optional[Sequence[LayerDefinition]] = None):
        if layers is None:
            layers = []
        return cls(layers, [], [], [], {})

    @classmethod
    def from_token_lists(
        cls, token_lists: Sequence[Sequence[str]], layers: Optional[Sequence[LayerDefinition]] = None
    ) -> 'Document':
        doc = Document.empty(layers)
        for tlist in token_lists:
            doc = doc.with_added_token_strs(tlist)
        return doc

    def token_sentence(self, token: Token) -> Sentence:
        return next(s for s in self.sentences if s.idx == token.sentence_idx)

    def annotation_sentences(self, annotation: AnnotationPart) -> List[Sentence]:
        return sorted({self.token_sentence(t) for t in annotation.tokens}, key=lambda s: s.idx)

    def sentence_tokens(self, sentence: Sentence) -> List[Token]:
        return [t for t in self.tokens if t.sentence_idx == sentence.idx]

    def match_annotations(self, sentence: Sentence = None, layer_name: str = '', field='') -> Sequence[AnnotationPart]:
        """
        Filter this document's annotations by the given criteria and return only those
        matching the given sentence, layer and field. Leave a parameter unfilled to
        include annotations with any value in that slot. For example:

            doc.match_annotations(layer='l1')

        returns annotations from layer 'l1' regardless of which sentence they are in or
        which field in that layer they have.
        """
        result = self.annotation_parts
        if sentence:
            result = [a for a in result if sentence in self.annotation_sentences(a)]
        if layer_name:
            result = [a for a in result if a.layer.name == layer_name]
        if field:
            result = [a for a in result if a.field == field]
        return result

    def with_added_token_strs(self, token_strs: Sequence[str]) -> 'Document':
        """
        Build a new document that contains a sentence made up of tokens from the token
        texts. This increments sentence and token indices and calculates (utf-16)
        offsets for the tokens as per the TSV standard.

        :param token_strs: The token texts to add.
        :return: A new document with the token strings added.
        """
        text = ' '.join(token_strs)
        sent_idx = len(self.sentences) + 1
        start = self.tokens[-1].end + 1 if self.tokens else 0
        tokens = tokens_from_strs(token_strs, sent_idx=sent_idx, token_start=start)
        sentence = Sentence(idx=sent_idx, text=text, tokens=tuple(tokens))

        return replace(self, sentences=[*self.sentences, sentence], tokens=[*self.tokens, *tokens])

    def tsv(self, linebreak='\n'):
        return webanno_tsv_write(self, linebreak)

    def get_layer(self, layer_name: str) -> LayerDefinition:
        return self._layers_by_name[layer_name]


def token_sort(tokens: Iterable[Token]) -> List[Token]:
    """
    Sort tokens by their sentence_idx first, then by the index in their sentence.
    """
    if not tokens:
        return []
    offset = max(t.idx for t in tokens) + 1
    return sorted(tokens, key=lambda t: (t.sentence_idx * offset) + t.idx)


def fix_annotation_ids(annotations: Iterable[AnnotationPart]) -> List[AnnotationPart]:
    """
    Setup label ids for the annotations to be consistent in the group.
    After this, there should be no duplicate label id and every multi-token
    annotation should have an id. Leaves present label_ids unchanged if possible.
    """
    with_ids = (a for a in annotations if a.label_id != NO_LABEL_ID)
    with_repeated_ids = {a for a in with_ids if a.label_id in [a2.label_id for a2 in with_ids if a2 != a]}
    without_ids = {a for a in annotations if len(a.tokens) > 1 and a.label_id == NO_LABEL_ID}
    both = without_ids.union(with_repeated_ids)
    if both:
        max_id = max((a.label_id for a in annotations), default=1)
        new_ids = itertools.count(max_id + 1)
        return [replace(a, label_id=next(new_ids)) if a in both else a for a in annotations]
    else:
        return list(annotations)


def utf_16_length(s: str) -> int:
    return int(len(s.encode('utf-16-le')) / 2)


def tokens_from_strs(token_strs: Sequence[str], sent_idx=1, token_start=0) -> [Token]:
    utf_16_lens = list(map(utf_16_length, token_strs))
    starts = [(sum(utf_16_lens[:i])) for i in range(len(utf_16_lens))]
    starts = [s + i for i, s in enumerate(starts)]  # offset the ' ' assumed between tokens
    starts = [s + token_start for s in starts]
    stops = [s + length for s, length in zip(starts, utf_16_lens)]
    return [Token(idx=i + 1, sentence_idx=sent_idx, start=s1, end=s2, text=t) for i, (s1, s2, t) in
            enumerate(zip(starts, stops, token_strs))]


def merge_into_annotations(annotations: Sequence[AnnotationPart], annotation: AnnotationPart) -> Sequence[AnnotationPart]:
    candidate = next((a for a in annotations if a.should_merge(annotation)), None)
    if candidate:
        merged = candidate.merge(annotation)
        return [a if a != candidate else merged for a in annotations]
    else:
        return [*annotations, annotation]


def _annotation_type(layer_name, field_name):
    return '|'.join([layer_name, field_name])


def _unescape(text: str) -> str:
    for s in RESERVED_STRS:
        text = text.replace('\\' + s, s)
    return text


def _escape(text: str) -> str:
    for s in RESERVED_STRS:
        text = text.replace(s, '\\' + s)
    return text


def _read_token(row: Dict) -> Token:
    """
    Construct a Token from the row object using the sentence from doc.
    This converts the first three columns of the TSV, e.g.:
        "2-3    13-20    example"
    becomes:
        Token(Sentence(idx=2), idx=3, start=13, end=20, text='example')
    """

    def intsplit(s: str):
        return [int(s) for s in s.split('-')]

    sent_idx, tok_idx = intsplit(row['sent_tok_idx'])
    start, end = intsplit(row['offsets'])
    text = _unescape(row['token'])
    return Token(sentence_idx=sent_idx, idx=tok_idx, start=start, end=end, text=text)


def _read_annotation_field(row: Dict, layer: LayerDefinition, field: str) -> List[str]:
    col_name = _annotation_type(layer.name, field)
    return row[col_name].split('|') if row[col_name] else []


def _read_label_and_id(field: str) -> Tuple[str, int]:
    """
    Reads a Webanno TSV field value, returning a label and an id.
    Returns an empty label for placeholder values '_', '*'
    Examples:
        "OBJ[6]" -> ("OBJ", 6)
        "OBJ"    -> ("OBJ", -1)
        "_"      -> ("", None)
        "*[6]"   -> ("", 6)
    """

    def handle_label(s: str):
        return '' if FIELD_EMPTY_RE.match(s) else _unescape(s)

    match = FIELD_WITH_ID_RE.match(field)
    if match:
        ## handle relation ids
        #if match.group(3) is not None:
        #    rel_target_id = int(match.group(3))
        #    # 0 indicates no disambiguation
        #    rel_id = NO_LABEL_ID if rel_target_id == 0 else rel_target_id
        #    return handle_label(match.group(1) + f"[{match.group(2)}]"), rel_id
        #else:
        #    return handle_label(match.group(1)), int(match.group(2))
        return handle_label(match.group(1)), int(match.group(2))
    else:
        return handle_label(field), NO_LABEL_ID


def _filter_sentences(lines: List[str]) -> List[str]:
    """
    Filter lines beginning with 'Text=', if multiple such lines are
    following each other, concatenate them.
    """
    matches = [SENTENCE_RE.match(line) for line in lines]
    match_groups = [list(ms) for is_m, ms in itertools.groupby(matches, key=lambda m: m is not None) if is_m]
    text_groups = [[m.group(1) for m in group] for group in match_groups]
    return [MULTILINE_SPLIT_CHAR.join(group) for group in text_groups]


def _tsv_read_lines(lines: List[str], overriding_layer_names: Dict[str, Sequence[LayerDefinition]] = None) -> Document:
    non_comments = [line for line in lines if not COMMENT_RE.match(line)]
    token_data = [line for line in non_comments if not SUB_TOKEN_RE.match(line)]
    sentence_strs = _filter_sentences(lines)

    if overriding_layer_names:
        layers = overriding_layer_names
    else:
        layers = LayerDefinition.from_lines(lines)

    columns = []
    for layer in layers:
        columns += layer.as_columns()
    rows = csv.DictReader(token_data, dialect=WebannoTsvDialect, fieldnames=TOKEN_FIELDNAMES + columns)

    annotation_parts = []
    tokens = []
    sentences = []
    sentence_tokens = []
    for row in rows:
        # consume the first three columns of each line
        token = _read_token(row)
        tokens.append(token)

        # if the sentence index changes, we have a new sentence
        if len(sentence_tokens) > 0 and token.sentence_idx != sentence_tokens[0].sentence_idx:
            sent_idx = sentence_tokens[0].sentence_idx
            sentence = Sentence(
                idx=sent_idx,
                text=sentence_strs[sent_idx - 1],
                tokens=tuple(sentence_tokens),
            )
            sentences.append(sentence)
            sentence_tokens = []

        sentence_tokens.append(token)

        # Each column after the first three is (part of) a span annotation layer
        for layer in layers:
            for annotation in layer.read_annotations(token, row):
                annotation_parts = merge_into_annotations(annotation_parts, annotation)

    # add the last sentence
    if len(sentence_tokens) > 0:
        sent_idx = sentence_tokens[0].sentence_idx
        sentence = Sentence(
            idx=sent_idx,
            text=sentence_strs[sent_idx - 1],
            tokens=tuple(sentence_tokens),
        )
        sentences.append(sentence)

    # we can have multiple annotations with the same annotation_id because it does not include the id for relations
    features_per_layer_and_id = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    tokens_per_layer_and_id = defaultdict(dict)
    for annotation_part in annotation_parts:
        features_per_layer_and_id[annotation_part.layer][annotation_part.annotation_id][annotation_part.field].append(annotation_part.label)
        tokens_per_layer_and_id[annotation_part.layer][annotation_part.annotation_id] = annotation_part.tokens

    annotations = defaultdict(list)
    for layer in layers:
        for annotation_id, feature_lists in features_per_layer_and_id[layer].items():
            annotation_tokens = tokens_per_layer_and_id[layer][annotation_id]
            for features in dict_of_lists_to_list_of_dicts(feature_lists):
                annotation = layer.new_annotation(
                    id=annotation_id, previous_annotations=annotations, tokens=annotation_tokens, **features
                )
                annotations[layer].append(annotation)

    return Document(
        layers=layers,
        sentences=sentences,
        tokens=tokens,
        annotation_parts=annotation_parts,
        annotations=annotations,
    )


def webanno_tsv_read_string(tsv: str, overriding_layer_def: List[Tuple[str, List[str]]] = None) -> Document:
    """
    Read the string content of a tsv file and return a Document representation

    :param tsv: The tsv input to read.
    :param overriding_layer_def: If this is given, use these names
        instead of headers defined in the tsv string to name layers
        and fields. See Document for an example of layers.
    :return: A Document instance of string input
    """
    return _tsv_read_lines(tsv.splitlines(), overriding_layer_def)


def webanno_tsv_read_file(path: str, overriding_layers: Optional[Sequence[LayerDefinition]] = None) -> Document:
    """
    Read the tsv file at path and return a Document representation.

    :param path: Path to read.
    :param overriding_layers: If this is given, use these names
        instead of headers defined in the file to name layers
        and fields. See Document for an example of layers.
    :return: A Document instance of the file at path.
    """
    with open(path, mode='r', encoding='utf-8') as f:
        lines = f.readlines()
    doc = _tsv_read_lines(lines, overriding_layers)
    return replace(doc, path=path)


def _write_label(label: Optional[str]):
    return _escape(label) if label else '*'


def _write_label_id(lid: int):
    return '' if lid == NO_LABEL_ID else '[%d]' % lid


def _write_label_and_id(label: Optional[str], label_id: int) -> str:
    return _write_label(label) + _write_label_id(label_id)


def _write_annotation_field(annotations_in_layer: Iterable[AnnotationPart], field: str) -> str:
    if not annotations_in_layer:
        return '_'

    with_field_val = {(a.label, a.label_id) for a in annotations_in_layer if a.field == field}

    all_ids = {a.label_id for a in annotations_in_layer if a.label_id != NO_LABEL_ID}
    ids_used = {label_id for _, label_id in with_field_val}
    without_field_val = {(None, label_id) for label_id in all_ids - ids_used}

    both = sorted(with_field_val.union(without_field_val), key=lambda t: t[1])
    labels = [_write_label_and_id(label, lid) for label, lid in both]
    if not labels:
        return '*'
    else:
        return '|'.join(labels)


def _write_sentence_header(text: str) -> List[str]:
    return ['', f'#Text={_escape(text)}']


def _write_token_fields(token: Token) -> Sequence[str]:
    return [
        f'{token.sentence_idx}-{token.idx}',
        f'{token.start}-{token.end}',
        _escape(token.text),
    ]


def _write_line(layers: Sequence[LayerDefinition], annotation_parts: List[AnnotationPart], token: Token) -> str:
    token_fields = _write_token_fields(token)
    layer_fields = []
    for layer in layers:
        annotations = [a for a in annotation_parts if a.layer == layer and token in a.tokens]
        layer_fields += [_write_annotation_field(annotations, field) for field in layer.fields]
    return '\t'.join([*token_fields, *layer_fields])


def webanno_tsv_write(doc: Document, linebreak='\n') -> str:
    """
    Return a tsv string that represents the given Document.
    If there are repeated label_ids in the Document's Annotations, these
    will be corrected. If there are Annotations that are missing a label_id,
    it will be added.
    """
    lines = []
    lines += HEADERS
    for layer in doc.layers:
        lines.append(layer.as_header())
    lines.append('')

    annotation_parts = fix_annotation_ids(doc.annotation_parts)

    for sentence in doc.sentences:
        lines += _write_sentence_header(sentence.text)
        for token in doc.sentence_tokens(sentence):
            lines.append(_write_line(layers=doc.layers, annotation_parts=annotation_parts, token=token))

    return linebreak.join(lines)
