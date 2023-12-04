import os
import unittest
from dataclasses import replace

from webanno_tsv.webanno_tsv import (
    webanno_tsv_read_file, webanno_tsv_read_string,
    AnnotationPart, Document, Sentence, Token,
    NO_LABEL_ID, SENTENCE_PADDING_CHAR, SpanLayer,
    _convert_annotation_parts_to_annotations,
)

# These are used to override the actual layer names in the test files for brevity
DEFAULT_LAYERS = [
    SpanLayer('l1', ('pos',)),
    SpanLayer('l2', ('lemma',)),
    SpanLayer('l3', ('entity_id', 'named_entity'))
]

ACTUAL_DEFAULT_LAYER_NAMES = [
    SpanLayer('de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.pos.POS', ('PosValue',)),
    SpanLayer('de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Lemma', ('value',)),
    SpanLayer('webanno.custom.LetterEntity', ('entity_id', 'value'))
]


def test_file(name):
    return os.path.join(os.path.dirname(__file__), 'resources', name)


class WebannoTsvModelTest(unittest.TestCase):

    def test_doc_tokens(self):
        strings = ['A', 'sentence', 'to', 'add', '.']
        doc = Document.from_token_lists([strings])
        self.assertEqual(5, len(doc.tokens))
        self.assertEqual(strings, [t.text for t in doc.tokens])


class WebannoTsvReadRegularFilesTest(unittest.TestCase):
    TEXT_SENT_1 = "929 Prof. Gerhard Braun an Gerhard Rom , 23 . Juli 1835 Roma li 23 Luglio 1835 ."
    TEXT_SENT_2 = "Von den anderen schönen Gefäßen dieser Entdeckungen führen " \
                  + "wir hier nur noch einen Kampf des Herkules mit dem Achelous auf ."

    def setUp(self) -> None:
        self.doc = webanno_tsv_read_file(test_file('test_input.tsv'), DEFAULT_LAYERS)

    def test_can_read_tsv(self):
        self.assertIsInstance(self.doc, Document)
        self.assertEqual(2, len(self.doc.sentences))

        fst, snd = self.doc.sentences
        self.assertIsInstance(fst, Sentence)
        self.assertIsInstance(snd, Sentence)

    def test_reads_correct_sentences(self):
        fst, snd = self.doc.sentences
        self.assertEqual(1, fst.idx)
        self.assertEqual(2, snd.idx)
        self.assertEqual(self.TEXT_SENT_1, fst.text)
        self.assertEqual(self.TEXT_SENT_2, snd.text)

    def test_reads_correct_document_text(self):
        text = SENTENCE_PADDING_CHAR.join((self.TEXT_SENT_1, self.TEXT_SENT_2))
        self.assertEqual(text, self.doc.text)

    def test_reads_correct_tokens(self):
        fst, snd = self.doc.sentences

        spot_checks = [(fst, 4, 18, 23, "Braun"),
                       (fst, 16, 67, 73, "Luglio"),
                       (snd, 1, 81, 84, "Von"),
                       (snd, 14, 164, 169, "Kampf"),
                       (snd, 21, 204, 205, ".")]
        for sentence, idx, start, end, text in spot_checks:
            token = self.doc.sentence_tokens(sentence)[idx - 1]
            self.assertEqual(idx, token.idx)
            self.assertEqual(start, token.start)
            self.assertEqual(end, token.end)
            self.assertEqual(text, token.text)

    def test_reads_correct_annotations(self):
        _, snd = self.doc.sentences

        poss = self.doc.match_annotations(snd, 'l1', 'pos')
        lemmas = self.doc.match_annotations(snd, 'l2', 'lemma')
        entity_ids = self.doc.match_annotations(snd, 'l3', 'entity_id')
        named_entities = self.doc.match_annotations(snd, 'l3', 'named_entity')

        self.assertEqual(21, len(poss))
        self.assertEqual(22, len(lemmas))
        self.assertEqual(0, len(entity_ids))
        self.assertEqual(3, len(named_entities))

        # some spot checks (first one, last one, some in between
        spot_checks = [(poss[0], 81, 84, 'APPR', -1, 'Von'),
                       (poss[3], 97, 104, 'ADJA', -1, 'schönen'),
                       (poss[11], 153, 157, 'ADV', -1, 'noch'),
                       (poss[20], 204, 205, '$.', -1, '.'),
                       (lemmas[0], 81, 84, 'von', -1, 'Von'),
                       (lemmas[3], 97, 104, 'schön', -1, 'schönen'),
                       (lemmas[12], 153, 157, 'noch', -1, 'noch'),
                       (lemmas[21], 204, 205, '.', -1, '.'),
                       (named_entities[0], 164, 199, 'OBJ', 8, 'Kampf des Herkules mit dem Achelous'),
                       (named_entities[1], 174, 182, 'PERmentioned', 9, 'Herkules'),
                       (named_entities[2], 191, 199, 'PERmentioned', 10, 'Achelous'),
                       ]
        for annotation, start, end, label, label_id, text in spot_checks:
            self.assertEqual(start, annotation.start)
            self.assertEqual(end, annotation.end)
            self.assertEqual(label, annotation.label)
            self.assertEqual(label_id, annotation.label_id)
            self.assertEqual(text, annotation.text)


class WebannoTsvReadFileWithFormatV33(unittest.TestCase):
    LAYERS = [
        SpanLayer('l1', ('entity_id', 'named_entity')),
        SpanLayer('l2', ('tex_layout',)),
    ]
    TEXT_SENT_1 = 'Braun an Gerhard Dresden, 10.'
    TEXT_SENT_2 = 'März 1832 (Zettel an den Brief geklebt) Die Intelligenzblätter habe ich zurückgehalten und sende ' \
                  + 'nur den Brief.\f' \
                  + '- Den Intelligenzblättern ist auf eine Anzeige der bald erscheinenden Bernhardischen Suidas ' \
                  + 'sowie der Scriptores hist., August, beigelegt.'

    def setUp(self) -> None:
        self.doc = webanno_tsv_read_file(test_file('test_input_v3.3.tsv'), self.LAYERS)

    def test_reads_doc(self):
        self.assertIsInstance(self.doc, Document)
        self.assertEqual(2, len(self.doc.sentences))
        self.assertEqual(self.TEXT_SENT_1, self.doc.sentences[0].text)

    def test_reads_multiline_sentence(self):
        self.assertEqual(self.TEXT_SENT_2, self.doc.sentences[1].text)

    def test_reads_annotations_correctly(self):
        self.assertEqual(9, len(self.doc.match_annotations(layer_name='l1', field='named_entity')))
        self.assertEqual(1, len(self.doc.match_annotations(layer_name='l2', field='tex_layout')))

        annotations = self.doc.match_annotations(layer_name='l1', field='named_entity')
        spot_checks = [
            (0, 'PERauthor', -1, 'Braun'),
            (3, 'DATEletter', 1, '10 . März 1832'),
            (8, 'LIT', 4, 'Scriptores hist . , August')
        ]
        for idx, label, label_id, text in spot_checks:
            self.assertEqual(label, annotations[idx].label)
            self.assertEqual(label_id, annotations[idx].label_id)
            self.assertEqual(text, annotations[idx].text)


class WebannoTsvReadActualFileHeaders(unittest.TestCase):

    def test_headers_from_input_file(self):
        doc = webanno_tsv_read_file(test_file('test_input.tsv'))
        self.assertEqual(ACTUAL_DEFAULT_LAYER_NAMES, doc.layers)


class WebannoTsvReadFileWithQuotesTest(unittest.TestCase):

    def test_reads_quotes(self):
        doc = webanno_tsv_read_file(test_file('test_input_quotes.tsv'), DEFAULT_LAYERS)
        tokens = doc.sentence_tokens(doc.sentences[0])

        self.assertEqual('\"', tokens[3].text)
        self.assertEqual('\"', tokens[5].text)
        self.assertEqual('quotes', tokens[4].text)


class WebannoTsvReadFileWithMultiSentenceSpanAnnotation(unittest.TestCase):

    def test_read_multi_sentence_annotation(self):
        self.doc = webanno_tsv_read_file(test_file('test_input_multi_sentence_span.tsv'), DEFAULT_LAYERS)

        annotations = self.doc.match_annotations(layer_name='l3', field='named_entity')
        self.assertEqual(1, len(annotations))

        annotation = annotations[0]
        self.assertEqual(66, annotation.label_id)
        self.assertEqual(2, len(annotation.tokens))
        self.assertEqual(['annotation-begin', 'annotation-end'], annotation.token_texts)


class WebannoAddTokensAsSentenceTest(unittest.TestCase):

    def test_add_simple(self):
        tokens = ['This', 'is', 'a', 'sentence', '.']
        doc = Document.from_token_lists([tokens])
        sentence = doc.sentences[0]

        self.assertIsInstance(sentence, Sentence)
        self.assertEqual(1, sentence.idx)
        self.assertEqual('This is a sentence .', sentence.text)
        self.assertEqual([sentence], doc.sentences)

        expected_tokens = [
            Token(sentence_idx=sentence.idx, idx=1, start=0, end=4, text='This'),
            Token(sentence_idx=sentence.idx, idx=2, start=5, end=7, text='is'),
            Token(sentence_idx=sentence.idx, idx=3, start=8, end=9, text='a'),
            Token(sentence_idx=sentence.idx, idx=4, start=10, end=18, text='sentence'),
            Token(sentence_idx=sentence.idx, idx=5, start=19, end=20, text='.'),
        ]
        self.assertEqual(expected_tokens, doc.sentence_tokens(sentence))

    def test_add_unicode_text(self):
        # Example from the WebAnno TSV docs. The smiley should increment
        # the offset by two as it counts for two chars in UTF-16 (as used by Java).
        tokens = ['I', 'like', 'it', '😊', '.']
        doc = Document.from_token_lists([tokens])

        self.assertEqual('😊', doc.tokens[3].text)
        self.assertEqual(10, doc.tokens[3].start)
        self.assertEqual(12, doc.tokens[3].end)
        self.assertEqual('.', doc.tokens[4].text)
        self.assertEqual(13, doc.tokens[4].start)


class WebannoTsvWriteTest(unittest.TestCase):

    def test_complete_writing(self):
        doc = Document.from_token_lists([
            ['First', 'sentence', '😊', '.'],
            ['Second', 'sentence', 'escape[t]his;token', '.']
        ], layers=DEFAULT_LAYERS)

        annotations = [
            AnnotationPart(tokens=tuple(doc.tokens[0:1]), layer=doc.get_layer('l1'), field='pos', label='pos-val'),
            AnnotationPart(tokens=tuple(doc.tokens[0:1]), layer=doc.get_layer('l2'), field='lemma', label='first'),
            AnnotationPart(tokens=tuple(doc.tokens[1:2]), layer=doc.get_layer('l2'), field='lemma', label='sentence'),
            AnnotationPart(tokens=tuple(doc.tokens[2:4]), layer=doc.get_layer('l3'), field='named_entity', label='smiley-end', label_id=37),
            AnnotationPart(tokens=tuple(doc.tokens[3:4]), layer=doc.get_layer('l3'), field='named_entity', label='DOT'),
            AnnotationPart(tokens=tuple(doc.tokens[7:8]), layer=doc.get_layer('l1'), field='pos', label='dot'),
            AnnotationPart(tokens=tuple(doc.tokens[5:6]), layer=doc.get_layer('l2'), field='lemma', label='sentence'),
            AnnotationPart(tokens=tuple(doc.tokens[7:8]), layer=doc.get_layer('l2'), field='lemma', label='.'),
            AnnotationPart(tokens=tuple(doc.tokens[4:5]), layer=doc.get_layer('l3'), field='named_entity', label='XYZ'),
            AnnotationPart(tokens=tuple(doc.tokens[6:7]), layer=doc.get_layer('l3'), field='named_entity', label='escape|this\\field'),
        ]
        annotations = _convert_annotation_parts_to_annotations(annotations, layers=doc.layers)
        doc = replace(doc, annotations=annotations)
        result = doc.tsv()

        expected = [
            '#FORMAT=WebAnno TSV 3.3',
            '#T_SP=l1|pos',
            '#T_SP=l2|lemma',
            '#T_SP=l3|entity_id|named_entity',
            '',
            '',
            '#Text=First sentence 😊 .',
            '1-1\t0-5\tFirst\tpos-val\tfirst\t_\t_',
            '1-2\t6-14\tsentence\t_\tsentence\t_\t_',
            '1-3\t15-17\t😊\t_\t_\t*[37]\tsmiley-end[37]',
            '1-4\t18-19\t.\t_\t_\t*[37]\tDOT|smiley-end[37]',
            '',
            '#Text=Second sentence escape\\[t\\]his\\;token .',
            '2-1\t20-26\tSecond\t_\t_\t*\tXYZ',
            '2-2\t27-35\tsentence\t_\tsentence\t_\t_',
            '2-3\t36-54\tescape\\[t\\]his\\;token\t_\t_\t*\tescape\\|this\\\\field',
            '2-4\t55-56\t.\tdot\t.\t_\t_'
        ]
        self.assertEqual(expected, result.split('\n'))

    def test_label_id_is_added_on_writing(self):
        doc = Document.from_token_lists([['A', 'B', 'C', 'D']], layers=DEFAULT_LAYERS)

        a_with_id = AnnotationPart(tokens=tuple(doc.tokens[1:3]), layer=doc.get_layer('l3'), field='named_entity', label='BC', label_id=67)
        a_without = AnnotationPart(tokens=tuple(doc.tokens[2:4]), layer=doc.get_layer('l3'), field='named_entity', label='CD')
        a_single_token = AnnotationPart(tokens=tuple(doc.tokens[3:4]), layer=doc.get_layer('l3'), field='named_entity', label='D')

        annotations = _convert_annotation_parts_to_annotations([a_with_id, a_without, a_single_token], layers=doc.layers)
        #annotations = {
        #    doc.get_layer('l3'): [
        #        SpanAnnotation(tokens=tuple(doc.tokens[1:3]), features=dict(named_entity='BC'), id="67"),
        #        SpanAnnotation(tokens=tuple(doc.tokens[2:4]), features=dict(named_entity='CD'), id="68"),
        #        SpanAnnotation(tokens=tuple(doc.tokens[3:4]), features=dict(named_entity='D'), id=str(NO_LABEL_ID)),
        #    ]
        #}
        doc = replace(doc, annotations=annotations)

        doc_new = webanno_tsv_read_string(doc.tsv())

        self.assertNotEqual(doc, doc_new)
        self.assertEqual(3, len(doc_new.annotation_parts))
        self.assertEqual(67, doc_new.annotation_parts[0].label_id)
        self.assertEqual(68, doc_new.annotation_parts[1].label_id,
                         'Should have added a new label id incremented from last max.')
        self.assertEqual(NO_LABEL_ID, doc_new.annotation_parts[2].label_id,
                         'Should not have added a new label id for a single token.')

    def test_read_write_equality(self):
        path = test_file('test_input.tsv')
        with open(path, encoding='utf8', mode='r') as f:
            content = f.read().rstrip()
        doc = webanno_tsv_read_file(path)
        self.assertEqual(content.splitlines(), doc.tsv().splitlines(),
                         'Output from file parsing should have the same lines as that file.')

        doc2 = webanno_tsv_read_string(content)
        self.assertEqual(content.splitlines(), doc2.tsv().splitlines(),
                         'Output from string parsing should have the same lines as that string.')
