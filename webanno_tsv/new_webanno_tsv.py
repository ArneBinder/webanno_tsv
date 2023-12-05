import abc
import csv
import itertools
import re
from collections import defaultdict
from copy import copy
from dataclasses import dataclass, replace, field
from typing import Dict, List, Optional, Sequence, Tuple, Any, Union, Iterator

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
RESERVED_STRS = ['\\', '[', ']', '|', '_', '->', ';', '\t', '\n', '*', '\r']

# Multiline sentences are split on this character per Webanno Appendix B
MULTILINE_SPLIT_CHAR = '\f'

# character used to pad text between sentences
SENTENCE_PADDING_CHAR = '\n'


def _unescape(text: str) -> str:
    for s in RESERVED_STRS:
        text = text.replace('\\' + s, s)
    return text


def _escape(text: str) -> str:
    for s in RESERVED_STRS:
        text = text.replace(s, '\\' + s)
    return text


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

    @staticmethod
    def from_row(row: Dict) -> 'RowToken':
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


@dataclass(frozen=True, order=True)
class RowAnnotation:
    sentence_idx: int
    token_idx: int
    data: Dict[str, str]

    @property
    def idx(self):
        return f"{self.sentence_idx}-{self.token_idx}"


@dataclass(frozen=True, order=True)
class Token:
    start: int
    end: int

    def row(self, sentence_idx: int, token_idx: int, sentence: 'Sentence') -> List[str]:
        token_text = self.text(sentence.text, offset=-sentence.start)
        return [f"{sentence_idx}-{token_idx}", f"{self.start}-{self.end}", token_text]

    def text(self, text: str, offset: int = 0) -> str:
        return text[self.start+offset:self.end+offset]


@dataclass(frozen=True)
class Sentence:
    text: str
    tokens: Tuple[Token, ...] = ()

    def add_token(self, token: Union[Token, RowToken], expected_text: Optional[str] = None) -> 'Sentence':
        if isinstance(token, RowToken):
            expected_text = token.text
            token = Token(start=token.start, end=token.end)
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
        return [''] + [f'#Text={text}' for text in self.text.split(MULTILINE_SPLIT_CHAR)]

    def annotation_lines(self, sentence_idx: int) -> List[List[str]]:

        result = []
        for idx, token in enumerate(self.tokens):
            current_row = token.row(sentence_idx=sentence_idx, token_idx=idx+1, sentence=self)
            result.append(current_row)
        return result


@dataclass(frozen=True)
class Annotation(abc.ABC):
    values: Tuple[Optional[str], ...]


@dataclass(frozen=True)
class SpanAnnotation(Annotation):
    tokens: Tuple[Token, ...]


@dataclass(frozen=True)
class RelationAnnotation(Annotation):
    source: SpanAnnotation
    target: SpanAnnotation


@dataclass(frozen=True)
class Layer(abc.ABC):
    definition: 'LayerDefinition'
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

    def sentence_annotation_lines(
            self, sentences: Sequence[Sentence], id2annotation: Dict[Tuple[str, str], Annotation]
    ) -> Iterator[List[List[str]]]:
        yield from self.definition.sentence_annotation_lines(self.annotations, sentences, id2annotation)


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
        return [f'{self.name}|{f}' for f in self.fields]

    @staticmethod
    def from_lines(lines: List[str]) -> List['LayerDefinition']:
        # TODO: do not change order of layers! This is not a problem at the moment, because relation layers
        #  are always defined after all span layers, but this might be a problem when adding chain layers.
        return SpanLayerDefinition.from_lines(lines) + RelationLayerDefinition.from_lines(lines)

    @abc.abstractmethod
    def sentence_annotation_lines(
        self,
        annotations: Sequence[Annotation],
        sentences: Sequence[Sentence],
        id2annotation: Dict[Tuple[str, str], Annotation],
    ) -> Iterator[List[List[str]]]:
        pass

    def build_layer(
        self,
        rows: List[RowAnnotation],
        id2annotation: Dict[Tuple[str, str], Annotation],
        sentences: Dict[int, 'Sentence'],
    ) -> 'Layer':
        annotations = self.build_annotations(rows=rows, id2annotation=id2annotation, sentences=sentences)
        return Layer(definition=self, annotations=tuple(annotations))

    def read_annotation_field(self, row: Dict, field_name: str) -> List[str]:
        col_name = '|'.join([self.name, field_name])
        return row[col_name].split('|') if row[col_name] else []

    def read_annotation_row(self, row: Dict, row_token: RowToken) -> List[RowAnnotation]:
        result = []
        layer_values = {f: self.read_annotation_field(row, f) for f in self.fields}
        for d in dict_of_lists_to_list_of_dicts(layer_values):
            filtered = {k: v for k, v in d.items() if v != '_'}
            if len(filtered) > 0:
                result.append(
                    RowAnnotation(
                        sentence_idx=row_token.sentence_idx, token_idx=row_token.idx, data=filtered
                    )
                )
        return result

    @abc.abstractmethod
    def build_annotations(
        self,
        rows: List[RowAnnotation],
        id2annotation: Dict[Tuple[str, str], Annotation],
        sentences: Dict[int, 'Sentence'],
    ) -> List[Annotation]:
        pass

    @staticmethod
    def write_annotation_features(annotation: Annotation, annotation_id: Optional[str] = None) -> List[str]:
        result = [_escape(v) if v is not None else "*" for v in annotation.values]
        if annotation_id is not None:
            result = [f"{entry}[{annotation_id}]" for entry in result]
        return result


@dataclass(frozen=True)
class SpanLayerDefinition(LayerDefinition):

    @staticmethod
    def from_lines(lines: List[str]) -> List['SpanLayerDefinition']:
        span_matches = [SPAN_LAYER_DEF_RE.match(line) for line in lines]
        layers = [
            SpanLayerDefinition(name=m.group(1), features=tuple(m.group(2).split('|'))) for m in span_matches if m
        ]
        return layers

    def as_header(self) -> str:
        name = self.name + '|' + '|'.join(self.features)
        return f'#T_SP={name}'

    def sentence_annotation_lines(
        self,
        annotations: Sequence[SpanAnnotation],
        sentences: Sequence[Sentence],
        id2annotation: Dict[Tuple[str, str], Annotation],
    ) -> Iterator[List[List[str]]]:
        max_id = 1
        annotation_indices_require_id = set()
        token2annotation_indices = defaultdict(list)
        for idx, annotation in enumerate(annotations):
            if len(annotation.tokens) > 1:
                annotation_indices_require_id.add(idx)
            for token in annotation.tokens:
                token2annotation_indices[token].append(idx)

        for token, indices in token2annotation_indices.items():
            if len(indices) > 1:
                annotation_indices_require_id.update(indices)

        annotation_idx2id = {}
        for sentence in sentences:
            sentence_rows = []
            for token in sentence.tokens:
                current_row_items = [[] for _ in range(len(self))]
                for annotation_idx in token2annotation_indices.get(token, []):
                    annotation_id = None
                    annotation = annotations[annotation_idx]
                    if annotation_idx in annotation_indices_require_id:
                        if annotation_idx not in annotation_idx2id:
                            annotation_idx2id[annotation_idx] = str(max_id)
                            id2annotation[(self.name, str(max_id))] = annotation
                            max_id += 1
                        annotation_id = annotation_idx2id[annotation_idx]
                    annotation_entries = self.write_annotation_features(annotation, annotation_id=annotation_id)
                    for i, entry in enumerate(annotation_entries):
                        current_row_items[i].append(entry)
                current_row = ['|'.join(item) if item else '_' for item in current_row_items]
                sentence_rows.append(current_row)
            yield sentence_rows

    @staticmethod
    def _read_label_and_id(field: str, default_id: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Reads a Webanno TSV field value, returning a label and an id.
        Returns an empty label for placeholder values '_', '*'
        Examples:
            "OBJ[6]" -> ("OBJ", "6)
            "OBJ"    -> ("OBJ", default_id)
            "_"      -> (None, None)
            "*[6]"   -> (None, 6)
        """

        def handle_label(s: str):
            return None if FIELD_EMPTY_RE.match(s) else _unescape(s)

        match = FIELD_WITH_ID_RE.match(field)
        if match:
            return handle_label(match.group(1)), match.group(2)
        else:
            value = handle_label(field)
            if value is None:
                return None, None
            else:
                return value, default_id

    def build_annotations(
        self,
        rows: List[RowAnnotation],
        id2annotation: Dict[Tuple[str, str], Annotation],
        sentences: Dict[int, 'Sentence'],
    ) -> List['SpanAnnotation']:
        merged_data = {}
        merged_tokens = defaultdict(list)
        for annotation_row in rows:
            lid = None
            d_without_id = {}
            for k, v in annotation_row.data.items():
                label, current_lid = self._read_label_and_id(field=v, default_id=annotation_row.idx)
                if lid is not None and current_lid != lid:
                    raise ValueError(f"Found multiple labels for the same annotation: {annotation_row.data}")
                lid = current_lid
                d_without_id[k] = label
            if lid is None:
                raise ValueError(f"Could not find id for annotation row: {annotation_row}")
            previous_data = merged_data.get(lid, d_without_id)
            if previous_data != d_without_id:
                raise ValueError(f"Found multiple labels for the same annotation: {d_without_id} != {previous_data}")
            merged_data[lid] = d_without_id
            token = sentences[annotation_row.sentence_idx-1].tokens[annotation_row.token_idx-1]
            merged_tokens[lid].append(token)

        annotations = []
        for k in merged_data:
            d = merged_data[k]
            values = tuple(d[f] for f in self.features)
            tokens = tuple(merged_tokens[k])
            annotations.append(SpanAnnotation(values=values, tokens=tokens))
            id2annotation[(self.name, k)] = annotations[-1]

        return annotations


@dataclass(frozen=True)
class RelationLayerDefinition(LayerDefinition):
    base: str

    def __len__(self):
        # +1 for the base field
        return len(self.features) + 1

    @property
    def base_field(self) -> str:
        return f"BT_{self.base}"

    @property
    def fields(self) -> Tuple[str, ...]:
        return self.features + (self.base_field,)

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
        name = '|'.join((self.name,) + self.features + (self.base_field,))
        return f'#T_RL={name}'

    def sentence_annotation_lines(
        self,
        annotations: Sequence[RelationAnnotation],
        sentences: Sequence[Sentence],
        id2annotation: Dict[Tuple[str, str], Annotation],
    ) -> Iterator[List[List[str]]]:

        annotation2id = {a: lid for (layer_name, lid), a in id2annotation.items() if layer_name == self.base}
        token_to_indices = {}
        for sentence_idx, sentence in enumerate(sentences):
            for token_idx, token in enumerate(sentence.tokens):
                token_to_indices[token] = f"{sentence_idx+1}-{token_idx+1}"

        annotations_per_token = defaultdict(list)
        for annotation in annotations:
            annotations_per_token[annotation.target.tokens[0]].append(annotation)

        for sentence_idx, sentence in enumerate(sentences):
            sentence_rows = []
            for token_idx, token in enumerate(sentence.tokens):
                current_row_items = [[] for _ in range(len(self))]
                for annotation in annotations_per_token.get(token, []):
                    annotation_entries = self.write_annotation_features(annotation)

                    source_token_indices = token_to_indices[annotation.source.tokens[0]]
                    source_id = annotation2id.get(annotation.source, "0")
                    target_id = annotation2id.get(annotation.target, "0")
                    source_entry = source_token_indices
                    if source_id != "0" or target_id != "0":
                        source_entry += f"[{source_id}_{target_id}]"
                    annotation_entries.append(source_entry)

                    for i, entry in enumerate(annotation_entries):
                        current_row_items[i].append(entry)

                current_row = ['|'.join(item) if item else '_' for item in current_row_items]
                sentence_rows.append(current_row)

            yield sentence_rows

    @staticmethod
    def _read_relation_source_and_target_idx(base_value: str, default_target_idx: str) -> Tuple[str, str]:
        match = RELATION_SOURCE_RE.match(base_value)
        if not match:
            raise ValueError(f"Could not parse relation source from {base_value}")

        if match.group(2) is None or match.group(2) == "0":
            source_idx = match.group(1)
        else:
            source_idx = match.group(2)
        if match.group(3) is None or match.group(3) == "0":
            target_idx = default_target_idx
        else:
            target_idx = match.group(3)
        return source_idx, target_idx

    def build_annotations(
        self,
        rows: List[RowAnnotation],
        id2annotation: Dict[Tuple[str, str], Annotation],
        sentences: Dict[int, 'Sentence'],
    ) -> List['RelationAnnotation']:
        annotations = []
        for annotation_row in rows:
            if set(annotation_row.data) != set(self.fields):
                raise ValueError(
                    f"Row {annotation_row.data} does not contain all fields of layer {self.name} ({self.fields})"
                )
            base_value = annotation_row.data[self.base_field]
            source_idx, target_idx = self._read_relation_source_and_target_idx(
                base_value=base_value, default_target_idx=annotation_row.idx
            )
            source: SpanAnnotation = id2annotation[(self.base, source_idx)]
            target: SpanAnnotation = id2annotation[(self.base, target_idx)]
            other_values = tuple(_unescape(annotation_row.data[f]) for f in self.features)
            annotation = RelationAnnotation(values=other_values, source=source, target=target)
            id2annotation[(self.name, annotation_row.idx)] = annotation
            annotations.append(annotation)

        return annotations


@dataclass(frozen=True)
class Document:
    _layers: Sequence[Layer]
    sentences: Sequence[Sentence]

    def __init__(self, _layers: Sequence[Layer], sentences: Sequence[Sentence] = None):
        object.__setattr__(self, '_layers', _layers)
        object.__setattr__(self, 'sentences', sentences or [])

    @property
    def layers(self) -> Dict[str, Layer]:
        return {layer.name: layer for layer in self._layers}

    def add_sentence(self, sentence: Sentence) -> 'Document':
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

    def add_annotations(self, layer_name: str, annotations: Sequence[Annotation]) -> 'Document':
        layer, idx = {l.name: (l, i) for i, l in enumerate(self._layers)}[layer_name]
        # verify that all features are satisfied
        for annotation in annotations:
            if not len(annotation.values) == len(layer.definition.features):
                raise ValueError(
                    f"Annotation {annotation} does not contain the same number of values as layer {layer_name} "
                    f"({layer.definition.features})"
                )

        new_layer = layer.extend(annotations)
        return replace(self, _layers=[*self._layers[:idx], new_layer, *self._layers[idx + 1:]])

    def add_annotation(self, layer_name: str, annotation: Annotation) -> 'Document':
        return self.add_annotations(layer_name, [annotation])

    def header_lines(self) -> List[str]:
        result = copy(HEADERS)
        for layer in self._layers:
            result.append(layer.definition.as_header())
        result.append('')
        return result

    def sentence_lines(self) -> List[str]:

        id2annotation = {}
        generators = [layer.sentence_annotation_lines(self.sentences, id2annotation) for layer in self._layers]
        for sentence_idx, sentence_and_annotations in enumerate(zip(self.sentences, *generators)):
            sentence = sentence_and_annotations[0]
            annotation_lines = sentence_and_annotations[1:]
            token_items = sentence.annotation_lines(sentence_idx + 1)
            result = sentence.header_lines()
            for token_and_layer_items in zip(token_items, *annotation_lines):
                # flatten the list of lists
                row_items = [item for sublist in token_and_layer_items for item in sublist]
                result.append('\t'.join(row_items))
            yield result

    def lines(self, linebreak='\n') -> List[str]:
        without_lb = self.header_lines() + [line for sentence in self.sentence_lines() for line in sentence]
        return [line + linebreak for line in without_lb]

    def tsv(self, linebreak='\n') -> str:
        return ''.join(self.lines(linebreak))

    @staticmethod
    def _filter_sentences(lines: List[str]) -> List[str]:
        """
        Filter lines beginning with 'Text=', if multiple such lines are
        following each other, concatenate them.
        """
        matches = [SENTENCE_RE.match(line) for line in lines]
        match_groups = [list(ms) for is_m, ms in itertools.groupby(matches, key=lambda m: m is not None) if is_m]
        text_groups = [[m.group(1) for m in group] for group in match_groups]
        return [MULTILINE_SPLIT_CHAR.join(group).replace("\\r", "\r") for group in text_groups]

    @classmethod
    def from_lines(cls, lines: List[str]) -> 'Document':
        non_comments = [line for line in lines if not COMMENT_RE.match(line)]
        token_data = [line for line in non_comments if not SUB_TOKEN_RE.match(line)]
        sentence_strs = cls._filter_sentences(lines)

        layer_definitions = LayerDefinition.from_lines(lines)

        columns = []
        for layer_def in layer_definitions:
            columns += layer_def.as_columns()
        rows = csv.DictReader(token_data, dialect=WebannoTsvDialect, fieldnames=TOKEN_FIELDNAMES + columns)

        annotation_rows = {layer_def: [] for layer_def in layer_definitions}
        sentences = {idx: Sentence(text=s) for idx, s in enumerate(sentence_strs)}
        row: Dict
        for row in rows:
            row_token = RowToken.from_row(row)
            sent_idx = row_token.sentence_idx - 1
            sentences[sent_idx] = sentences[sent_idx].add_token(token=row_token)
            for layer_def in layer_definitions:
                annotation_dicts = layer_def.read_annotation_row(row=row, row_token=row_token)
                annotation_rows[layer_def].extend(annotation_dicts)

        id2annotation = {}
        layers = [
            layer_def.build_layer(
                rows=annotation_rows[layer_def], id2annotation=id2annotation, sentences=sentences
            )
            for layer_def in layer_definitions
        ]
        return cls(_layers=layers, sentences=tuple(sentences.values()))

    @classmethod
    def from_file(cls, path: str) -> 'Document':
        with open(path, 'r', encoding='utf-8') as f:
            return cls.from_lines(f.readlines())
