import abc
import csv
import itertools
import re
from collections import defaultdict
from copy import copy
from dataclasses import dataclass, replace, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Any, Callable

NO_LABEL_ID = -1

COMMENT_RE = re.compile('^#')
SPAN_LAYER_DEF_RE = re.compile(r'^#T_SP=([^|]+)\|(.*)$')
RELATION_LAYER_DEF_RE = re.compile(r'^#T_RL=([^|]+)\|(.*)$')
RELATION_BASE_LAYER = re.compile(r'^BT_(.+)$')
SENTENCE_RE = re.compile('^#Text=(.*)')
FIELD_EMPTY_RE = re.compile('^[_*]')
FIELD_WITH_ID_RE = re.compile(r'(.*)\[([0-9]*)]$')
SUB_TOKEN_RE = re.compile(r'[0-9]+-[0-9]+\.[0-9]+')
RELATION_SOURCE_RE = re.compile(r'^([0-9]+-[0-9]+)(?:\[([0-9]+)_([0-9]+)\])?$')

TOKEN_ID_RE = re.compile(r'^([0-9]+)-([0-9]+)$')

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
class RowToken:
    sentence_idx: int
    idx: int
    start: int
    end: int
    text: str



@dataclass(frozen=True)
class Annotation(abc.ABC):
    values: Tuple[Any, ...]


@dataclass(frozen=True, order=True)
class LayerDefinition(abc.ABC):
    name: str
    features: Tuple[str, ...]

    @abc.abstractmethod
    def as_header(self) -> str:
        pass

    def __len__(self) -> int:
        return len(self.features)

    @property
    def fields(self) -> Tuple[str, ...]:
        return self.features

    def as_columns(self) -> List[str]:
        return [f'{self.name}|{field}' for field in self.fields]

    @staticmethod
    def from_lines(lines: List[str]) -> List['LayerDefinition']:
        return SpanLayerDefinition.from_lines(lines) + RelationLayerDefinition.from_lines(lines)

    def new_layer(self, dicts: List[Dict[str, Tuple[str, int]]], annotation_to_id: Dict[Annotation, int], sentences: List['NewSentence']) -> 'Layer':

        if isinstance(self, SpanLayerDefinition):
            return SpanLayer.from_dicts(definition=self, dicts=dicts, annotation_to_id=annotation_to_id, sentences=sentences)
        elif isinstance(self, RelationLayerDefinition):
            return RelationLayer.from_dicts(definition=self, dicts=dicts, annotation_to_id=annotation_to_id, sentences=sentences)
        else:
            raise ValueError(f"Unknown layer definition type {type(self)}")

    def read_annotation_row(self, row: Dict, token_idx: int, sentence_idx: int) -> List[Dict[str, Tuple[str, int]]]:
        result = []
        layer_values = {field: _read_annotation_field(row, self, field) for field in self.fields}
        for d in dict_of_lists_to_list_of_dicts(layer_values):
            values_with_id = {k: _read_label_and_id(v) for k, v in d.items()}
            filtered = {k: (label, lid) for k, (label, lid) in values_with_id.items() if label != ''}
            if len(filtered) > 0:
                filtered["token_idx"] = token_idx
                filtered["sentence_idx"] = sentence_idx
                result.append(filtered)
        return result


@dataclass(frozen=True)
class SpanLayerDefinition(LayerDefinition):

    @staticmethod
    def from_lines(lines: List[str]) -> List['SpanLayerDefinition']:
        span_matches = [SPAN_LAYER_DEF_RE.match(line) for line in lines]
        layers = [SpanLayerDefinition(name=m.group(1), features=tuple(m.group(2).split('|'))) for m in span_matches if m]
        return layers


    def as_header(self) -> str:
        """
            Example:
                ('one', ['x', 'y', 'z']) => '#T_SP=one|x|y|z'
            """
        name = self.name + '|' + '|'.join(self.features)
        return f'#T_SP={name}'


@dataclass(frozen=True)
class RelationLayerDefinition(LayerDefinition):
    base: str

    def __len__(self):
        # +1 for the base field
        return len(self.features) + 1

    @property
    def fields(self) -> Tuple[str, ...]:
        return self.features + (f"BT_{self.base}",)

    @staticmethod
    def from_lines(lines: List[str]) -> List['RelationLayerDefinition']:
        relation_matches = [RELATION_LAYER_DEF_RE.match(line) for line in lines]
        layers = []
        for m in relation_matches:
            if m:
                name = m.group(1)
                features_and_base = m.group(2).split('|')
                base_match = RELATION_BASE_LAYER.match(features_and_base[-1])
                if not base_match:
                    raise ValueError(f"Could not parse base layer from {features_and_base[-1]}")
                layers.append(RelationLayerDefinition(
                    name=name, features=tuple(features_and_base[:-1]), base=base_match.group(1))
                )
        return layers


    def as_header(self) -> str:
        """
            Example:
                ('one', ['x', 'y', 'z']) => '#T_RL=one|x|y|z'
            """
        name = '|'.join((self.name,) + self.features + (f"BT_{self.base}",))
        return f'#T_RL={name}'


@dataclass(frozen=True, order=True)
class NewToken:
    start: int
    end: int

    def row(self, sentence_idx: int, token_idx: int, sentence: 'NewSentence') -> List[str]:
        token_text = self.text(sentence.text, offset=-sentence.start)
        return [f"{sentence_idx}-{token_idx}", f"{self.start}-{self.end}", token_text]

    def text(self, text: str, offset: int = 0) -> str:
        return text[self.start+offset:self.end+offset]


@dataclass(frozen=True)
class NewSentence:
    text: str
    tokens: Tuple[NewToken, ...] = ()

    def add_token(self, token: NewToken, expected_text: Optional[str] = None) -> 'NewSentence':
        new_sentence = replace(self, tokens=tuple([*self.tokens, token]))
        if expected_text is not None:
            token_text = token.text(new_sentence.text, offset=-new_sentence.start)
            if token_text != expected_text:
                raise ValueError(
                    f"Offset based token text '{token_text}' does not match the expected text '{expected_text}'"
                )
        return new_sentence

    @property
    def start(self):
        return self.tokens[0].start

    @property
    def end(self):
        return self.tokens[-1].end

    def header_lines(self) -> List[str]:
        return [f'#Text={text}' for text in self.text.split(MULTILINE_SPLIT_CHAR)]

    def annotation_lines(self, sentence_idx: int) -> List[List[str]]:

        result = []
        for idx, token in enumerate(self.tokens):
            current_row = token.row(sentence_idx=sentence_idx, token_idx=idx+1, sentence=self)
            result.append(current_row)
        return result


@dataclass(frozen=True)
class NewSpanAnnotation(Annotation):
    tokens: Tuple[NewToken, ...]


@dataclass(frozen=True)
class NewRelationAnnotation(Annotation):
    source: NewSpanAnnotation
    target: NewSpanAnnotation


@dataclass(frozen=True)
class Layer(abc.ABC):
    definition: LayerDefinition
    annotations: Tuple[Annotation, ...] = ()

    def append(self, annotation: Annotation) -> 'Layer':
        return replace(self, annotations=tuple([*self.annotations, annotation]))

    def extend(self, annotations: Sequence[Annotation]) -> 'Layer':
        return replace(self, annotations=tuple([*self.annotations, *annotations]))

    @property
    def name(self):
        return self.definition.name

    def __getitem__(self, item):
        return self.annotations[item]

    def __len__(self):
        return len(self.annotations)

    @abc.abstractmethod
    def sentence_annotation_lines(
            self, sentences: Sequence[NewSentence], annotation_to_id: Dict[Annotation, int]
    ) -> List[List[str]]:
        pass

    def format_annotation_features(
        self, annotation: 'Annotation', id: Optional[str] = None
    ) -> List[str]:
        result = [_escape(v) if v is not None else "*" for v in annotation.values]
        if id is not None:
            result = [f"{entry}[{id}]" for entry in result]
        return result

    @classmethod
    def from_dicts(
        cls, definition, dicts: List[Dict[str, Tuple[str, int]]], annotation_to_id: Dict[Annotation, int],
        sentences: List['NewSentence']
    ) -> 'Layer':
        raise NotImplementedError()


@dataclass(frozen=True)
class SpanLayer(Layer):

    def sentence_annotation_lines(
        self, sentences: Sequence[NewSentence], annotation_to_id: Dict[Annotation, int]
    ) -> List[List[str]]:
        max_id = 1
        annotations_per_token = defaultdict(list)
        annotation: NewSpanAnnotation
        for annotation in self.annotations:
            for token in annotation.tokens:
                annotations_per_token[token].append(annotation)

        annotations_require_id = set()
        for token, annotations in annotations_per_token.items():
            if len(annotations) > 1:
                annotations_require_id.update(annotations)

        for sentence in sentences:
            sentence_rows = []
            for token in sentence.tokens:
                current_row_items = [[] for _ in range(len(self.definition))]
                for annotation in annotations_per_token.get(token, []):
                    annotation_id = None
                    if annotation in annotations_require_id:
                        if annotation not in annotation_to_id:
                            annotation_to_id[annotation] = max_id
                            max_id += 1
                        annotation_id = annotation_to_id[annotation]
                    annotation_entries = self.format_annotation_features(annotation, id=annotation_id)
                    for i, entry in enumerate(annotation_entries):
                        current_row_items[i].append(entry)
                current_row = ['|'.join(item) if item else '_' for item in current_row_items]
                sentence_rows.append(current_row)
            yield sentence_rows


@dataclass(frozen=True)
class RelationLayer(Layer):

    def sentence_annotation_lines(
        self, sentences: Sequence[NewSentence], annotation_to_id: Dict[Annotation, int]
    ) -> List[List[str]]:

        token_to_indices = {}
        for sentence_idx, sentence in enumerate(sentences):
            for token_idx, token in enumerate(sentence.tokens):
                token_to_indices[token] = f"{sentence_idx+1}-{token_idx+1}"

        annotations_per_token = defaultdict(list)
        annotation: NewRelationAnnotation
        for annotation in self.annotations:
            annotations_per_token[annotation.target.tokens[0]].append(annotation)

        for sentence_idx, sentence in enumerate(sentences):
            sentence_rows = []
            for token_idx, token in enumerate(sentence.tokens):
                current_row_items = [[] for _ in range(len(self.definition))]
                for annotation in annotations_per_token.get(token, []):
                    annotation_entries = self.format_annotation_features(annotation)

                    source_token_indices = token_to_indices[annotation.source.tokens[0]]
                    source_id = annotation_to_id.get(annotation.source, 0)
                    target_id = annotation_to_id.get(annotation.target, 0)
                    source_entry = source_token_indices
                    if source_id != 0 or target_id != 0:
                        source_entry += f"[{source_id}_{target_id}]"
                    annotation_entries.append(source_entry)

                    for i, entry in enumerate(annotation_entries):
                        current_row_items[i].append(entry)

                current_row = ['|'.join(item) if item else '_' for item in current_row_items]
                sentence_rows.append(current_row)

            yield sentence_rows


@dataclass(frozen=True)
class NewDocument:
    _layers: Sequence[Layer]
    sentences: Sequence[NewSentence]

    def __init__(self, _layers: Sequence[Layer], sentences: Sequence[NewSentence] = None):
        object.__setattr__(self, '_layers', _layers)
        object.__setattr__(self, 'sentences', sentences or [])

    @property
    def layers(self) -> Dict[str, Layer]:
        return {layer.name: layer for layer in self._layers}

    def add_sentence(self, sentence: NewSentence) -> 'NewDocument':
        return replace(self, sentences=[*self.sentences, sentence])

    @property
    def tokens(self):
        return tuple([token for sentence in self.sentences for token in sentence.tokens])

    @property
    def text(self):
        # we need to use the sentence offsets to reconstruct padding in between them
        result = ''
        for sentence in self.sentences:
            result += SENTENCE_PADDING_CHAR * (sentence.start - len(result))
            result += sentence.text
        return result

    def add_annotations(self, layer_name: str, annotations: Sequence[Annotation]) -> 'NewDocument':
        layer, idx = {l.name: (l, i) for i, l in enumerate(self._layers)}[layer_name]
        # verify that all features are satisfied
        for annotation in annotations:
            #if not all([f in annotation.features for f in layer.definition.features]):
            #    raise ValueError(
            #        f"Annotation {annotation} does not contain all features of layer {layer_name} "
            #        f"({layer.definition.features})"
            #    )
            if not len(annotation.values) == len(layer.definition.features):
                raise ValueError(
                    f"Annotation {annotation} does not contain the same number of values as layer {layer_name} "
                    f"({layer.definition.features})"
                )

        new_layer = layer.extend(annotations)
        return replace(self, _layers=[*self._layers[:idx], new_layer, *self._layers[idx + 1:]])

    def add_annotation(self, layer_name: str, annotation: Annotation) -> 'NewDocument':
        return self.add_annotations(layer_name, [annotation])

    def header_lines(self) -> List[str]:
        result = copy(HEADERS)
        for layer in self._layers:
            result.append(layer.definition.as_header())
        result.append('')
        result.append('')
        return result

    def sentence_lines(self) -> List[str]:

        annotation_to_id = {}
        generators = [layer.sentence_annotation_lines(self.sentences, annotation_to_id) for layer in self._layers]
        for sentence_idx, sentence_and_annotations in enumerate(zip(self.sentences, *generators)):
            sentence = sentence_and_annotations[0]
            annotation_lines = sentence_and_annotations[1:]
            token_items = sentence.annotation_lines(sentence_idx + 1)
            result = sentence.header_lines()
            for token_and_layer_items in zip(token_items, *annotation_lines):
                # flatten the list of lists
                row_items = [item for sublist in token_and_layer_items for item in sublist]
                result.append('\t'.join(row_items))
            result.append('')
            yield result

    def tsv(self, linebreak='\n'):
        lines = self.header_lines() + [line for sentence in self.sentence_lines() for line in sentence]
        return linebreak.join(lines)

    @classmethod
    def from_lines(cls, lines: List[str]) -> 'NewDocument':
        non_comments = [line for line in lines if not COMMENT_RE.match(line)]
        token_data = [line for line in non_comments if not SUB_TOKEN_RE.match(line)]
        sentence_strs = _filter_sentences(lines)

        layer_definitions = LayerDefinition.from_lines(lines)

        columns = []
        for layer_def in layer_definitions:
            columns += layer_def.as_columns()
        rows = csv.DictReader(token_data, dialect=WebannoTsvDialect, fieldnames=TOKEN_FIELDNAMES + columns)

        annotations = {layer_def: [] for layer_def in layer_definitions}
        tokens = []
        sentences = {idx + 1: NewSentence(text=s) for idx, s in enumerate(sentence_strs)}
        for row in rows:
            # consume the first three columns of each line
            token = _read_token(row)
            tokens.append(token)

            sent_idx = token.sentence_idx
            sentences[sent_idx] = sentences[sent_idx].add_token(NewToken(start=token.start, end=token.end), token.text)
            for layer_def in layer_definitions:
                annotation_dicts = layer_def.read_annotation_row(row, token_idx=token.idx, sentence_idx=sent_idx)
                if annotation_dicts:
                    annotations[layer_def].extend(annotation_dicts)

        annotation_to_id = {}
        layers = [layer_def.new_layer(annotations[layer_def], annotation_to_id, sentences) for layer_def in layer_definitions]
        return cls(_layers=layers, sentences=tuple(sentences.values()))

    @classmethod
    def from_file(cls, path: str) -> 'NewDocument':
        with open(path, 'r', encoding='utf-8') as f:
            return cls.from_lines(f.readlines())


def token_sort(tokens: Iterable[RowToken]) -> List[RowToken]:
    """
    Sort tokens by their sentence_idx first, then by the index in their sentence.
    """
    if not tokens:
        return []
    offset = max(t.idx for t in tokens) + 1
    return sorted(tokens, key=lambda t: (t.sentence_idx * offset) + t.idx)


def utf_16_length(s: str) -> int:
    return int(len(s.encode('utf-16-le')) / 2)


def tokens_from_strs(token_strs: Sequence[str], sent_idx=1, token_start=0) -> [RowToken]:
    utf_16_lens = list(map(utf_16_length, token_strs))
    starts = [(sum(utf_16_lens[:i])) for i in range(len(utf_16_lens))]
    starts = [s + i for i, s in enumerate(starts)]  # offset the ' ' assumed between tokens
    starts = [s + token_start for s in starts]
    stops = [s + length for s, length in zip(starts, utf_16_lens)]
    return [RowToken(idx=i + 1, sentence_idx=sent_idx, start=s1, end=s2, text=t) for i, (s1, s2, t) in
            enumerate(zip(starts, stops, token_strs))]


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


def _read_token(row: Dict) -> RowToken:
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
    return RowToken(sentence_idx=sent_idx, idx=tok_idx, start=start, end=end, text=text)


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

