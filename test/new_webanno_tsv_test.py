import os
import unittest

from webanno_tsv.new_webanno_tsv import (
    RowToken,
    NO_LABEL_ID, SENTENCE_PADDING_CHAR, SpanLayerDefinition, Document, SpanLayer, RelationLayerDefinition, RelationLayer,
    Sentence, Token, SpanAnnotation, RelationAnnotation,
)

# These are used to override the actual layer names in the test files for brevity
DEFAULT_LAYERS = [
    SpanLayerDefinition('l1', ('pos',)),
    SpanLayerDefinition('l2', ('lemma',)),
    SpanLayerDefinition('l3', ('entity_id', 'named_entity'))
]

ACTUAL_DEFAULT_LAYER_NAMES = [
    SpanLayerDefinition('de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.pos.POS', ('PosValue',)),
    SpanLayerDefinition('de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Lemma', ('value',)),
    SpanLayerDefinition('webanno.custom.LetterEntity', ('entity_id', 'value'))
]


def test_file(name):
    return os.path.join(os.path.dirname(__file__), 'resources', name)


class WebannoTsvCreateDocumentFromScratchEmpty(unittest.TestCase):

    def setUp(self) -> None:
        layers = [
            SpanLayer(definition=SpanLayerDefinition('pos', ('pos',))),
            SpanLayer(definition=SpanLayerDefinition('lemma', ('lemma',))),
            RelationLayer(definition=RelationLayerDefinition('relations', ('label', 'trigger'), base='pos')),
        ]
        self.doc = Document(layers)

    def test_empty_doc(self):
        self.assertIsInstance(self.doc, Document)
        self.assertEqual(0, len(self.doc.sentences))
        self.assertEqual(0, len(self.doc.tokens))
        self.assertEqual("", self.doc.text)


class WebannoTsvCreateDocumentFromScratchWithAnnotations(unittest.TestCase):

    def setUp(self) -> None:
        layers = [
            SpanLayer(definition=SpanLayerDefinition('pos', ('pos',))),
            SpanLayer(definition=SpanLayerDefinition('lemma', ('lemma',))),
            RelationLayer(definition=RelationLayerDefinition('relations', ('label', 'trigger'), base='pos')),
        ]
        doc = Document(layers)

        first_sentence = (
            Sentence(text="This is a sentence.")
            .add_token(Token(start=0, end=4), expected_text='This')
            .add_token(Token(start=5, end=7), expected_text='is')
            .add_token(Token(start=8, end=9), expected_text='a')
            .add_token(Token(start=10, end=18), expected_text='sentence')
            .add_token(Token(start=18, end=19), expected_text='.')
        )
        # Add a second sentence with an offset to test that the sentence padding char is added correctly
        sent_offset = 21
        doc = doc.add_sentence(first_sentence)
        second_sentence = (
            Sentence(text="This is \fanother sentence.")
            .add_token(Token(start=0 + sent_offset, end=4 + sent_offset), expected_text='This')
            .add_token(Token(start=5 + sent_offset, end=7 + sent_offset), expected_text='is')
            .add_token(Token(start=9 + sent_offset, end=16 + sent_offset), expected_text='another')
            .add_token(Token(start=17 + sent_offset, end=25 + sent_offset), expected_text='sentence')
            .add_token(Token(start=25 + sent_offset, end=26 + sent_offset), expected_text='.')
        )
        doc = doc.add_sentence(second_sentence)

        # add annotations
        doc = doc.add_annotations(
            # add span annotations with some overlap
            layer_name='pos', annotations=[
                SpanAnnotation(tokens=doc.tokens[2:3], values=('article',)),
                SpanAnnotation(tokens=doc.tokens[2:4], values=('DT',)),
                SpanAnnotation(tokens=doc.tokens[7:8], values=('DT',)),
            ]
        )
        doc = doc.add_annotations(
            layer_name='lemma', annotations=[
                SpanAnnotation(tokens=doc.tokens[1:2], values=('be',)),
                SpanAnnotation(tokens=doc.tokens[6:7], values=('be',)),
            ]
        )
        doc = doc.add_annotation(
            layer_name='relations', annotation=RelationAnnotation(
                values=('same_type', "another"),
                source=doc.layers['pos'][1],
                target=doc.layers['pos'][2],
            )
        )
        self.doc = doc

    def test_add_sentences_and_annotations(self):
        self.assertEqual("This is a sentence.\n\nThis is \fanother sentence.", self.doc.text)
        self.assertEqual(2, len(self.doc.sentences))
        self.assertEqual(10, len(self.doc.tokens))

        self.assertEqual({"pos", "lemma", "relations"}, set(self.doc.layers))
        self.assertEqual(3, len(self.doc.layers['pos']))
        self.assertEqual(self.doc.layers['pos'][0], SpanAnnotation(tokens=self.doc.tokens[2:3], values=('article',)))
        self.assertEqual(self.doc.layers['pos'][1], SpanAnnotation(tokens=self.doc.tokens[2:4], values=('DT',)))
        self.assertEqual(self.doc.layers['pos'][2], SpanAnnotation(tokens=self.doc.tokens[7:8], values=('DT',)))
        self.assertEqual(2, len(self.doc.layers['lemma']))
        self.assertEqual(self.doc.layers['lemma'][0], SpanAnnotation(tokens=self.doc.tokens[1:2], values=('be',)))
        self.assertEqual(self.doc.layers['lemma'][1], SpanAnnotation(tokens=self.doc.tokens[6:7], values=('be',)))
        self.assertEqual(1, len(self.doc.layers['relations']))
        self.assertEqual(self.doc.layers['relations'][0], RelationAnnotation(
            values=('same_type', "another"),
            source=self.doc.layers['pos'][1],
            target=self.doc.layers['pos'][2],
        ))

    def test_layer_header(self):
        self.assertEqual(
            [
                '#FORMAT=WebAnno TSV 3.3',
                '#T_SP=pos|pos',
                '#T_SP=lemma|lemma',
                '#T_RL=relations|label|trigger|BT_pos',
                '',
                '',
            ],
            self.doc.header_lines()
        )

    def test_sentence_header_lines(self):
        self.assertEqual(2, len(self.doc.sentences))
        self.assertEqual(['#Text=This is a sentence.'], self.doc.sentences[0].header_lines())
        self.assertEqual(['#Text=This is ', '#Text=another sentence.'], self.doc.sentences[1].header_lines())

    def test_sentence_annotation_lines(self):
        self.assertEqual(2, len(self.doc.sentences))
        annotation_lines_sentence1 = self.doc.sentences[0].annotation_lines(sentence_idx=1)
        self.assertEqual([
            ['1-1', '0-4', 'This'],
            ['1-2', '5-7', 'is'],
            ['1-3', '8-9', 'a'],
            ['1-4', '10-18', 'sentence'],
            ['1-5', '18-19', '.'],
        ], annotation_lines_sentence1)
        annotation_lines_sentence2 = self.doc.sentences[1].annotation_lines(sentence_idx=2)
        self.assertEqual([
            ['2-1', '21-25', 'This'],
            ['2-2', '26-28', 'is'],
            ['2-3', '30-37', 'another'],
            ['2-4', '38-46', 'sentence'],
            ['2-5', '46-47', '.'],
        ], annotation_lines_sentence2)

    def test_layer_annotation_lines(self):
        self.assertEqual(3, len(self.doc.layers))
        annotation_to_id = {}
        pos_annotation_lines = list(self.doc.layers["pos"].sentence_annotation_lines(
            sentences=self.doc.sentences, annotation_to_id=annotation_to_id
        ))
        self.assertEqual([
            [
                ['_'],
                ['_'],
                ['article[1]|DT[2]'],
                ['DT[2]'],
                ['_'],
            ], [
                ['_'],
                ['_'],
                ['DT'],
                ['_'],
                ['_'],
            ]
        ], pos_annotation_lines)
        lemma_annotation_lines = list(self.doc.layers["lemma"].sentence_annotation_lines(
            sentences=self.doc.sentences, annotation_to_id=annotation_to_id
        ))
        self.assertEqual([
            [
                ['_'],
                ['be'],
                ['_'],
                ['_'],
                ['_'],
            ], [
                ['_'],
                ['be'],
                ['_'],
                ['_'],
                ['_'],
            ]
        ], lemma_annotation_lines)
        relation_annotation_lines = list(self.doc.layers["relations"].sentence_annotation_lines(
            sentences=self.doc.sentences, annotation_to_id=annotation_to_id
        ))
        self.assertEqual([
            [
                ['_', '_', '_'],
                ['_', '_', '_'],
                ['_', '_', '_'],
                ['_', '_', '_'],
                ['_', '_', '_'],
            ], [
                ['_', '_', '_'],
                ['_', '_', '_'],
                ['same\_type', 'another', '1-3[2_0]'],
                ['_', '_', '_'],
                ['_', '_', '_'],
            ]
        ], relation_annotation_lines)

    def test_sentence_lines(self):
        lines = list(self.doc.sentence_lines())
        self.assertEqual(2, len(lines))
        self.assertEqual([
            '#Text=This is a sentence.',
            '1-1\t0-4\tThis\t_\t_\t_\t_\t_',
            '1-2\t5-7\tis\t_\tbe\t_\t_\t_',
            '1-3\t8-9\ta\tarticle[1]|DT[2]\t_\t_\t_\t_',
            '1-4\t10-18\tsentence\tDT[2]\t_\t_\t_\t_',
            '1-5\t18-19\t.\t_\t_\t_\t_\t_',
            '',
        ], lines[0])
        self.assertEqual([
            '#Text=This is ',
            '#Text=another sentence.',
            '2-1\t21-25\tThis\t_\t_\t_\t_\t_',
            '2-2\t26-28\tis\t_\tbe\t_\t_\t_',
            '2-3\t30-37\tanother\tDT\t_\tsame\_type\tanother\t1-3[2_0]',
            '2-4\t38-46\tsentence\t_\t_\t_\t_\t_',
            '2-5\t46-47\t.\t_\t_\t_\t_\t_',
            '',
        ], lines[1])

    def test_tsv(self):
        tsv = self.doc.tsv()
        expected = (
            '#FORMAT=WebAnno TSV 3.3\n'
            '#T_SP=pos|pos\n'
            '#T_SP=lemma|lemma\n'
            '#T_RL=relations|label|trigger|BT_pos\n'
            '\n'
            '\n'
            '#Text=This is a sentence.\n'
            '1-1\t0-4\tThis\t_\t_\t_\t_\t_\n'
            '1-2\t5-7\tis\t_\tbe\t_\t_\t_\n'
            '1-3\t8-9\ta\tarticle[1]|DT[2]\t_\t_\t_\t_\n'
            '1-4\t10-18\tsentence\tDT[2]\t_\t_\t_\t_\n'
            '1-5\t18-19\t.\t_\t_\t_\t_\t_\n'
            '\n'
            '#Text=This is \n'
            '#Text=another sentence.\n'
            '2-1\t21-25\tThis\t_\t_\t_\t_\t_\n'
            '2-2\t26-28\tis\t_\tbe\t_\t_\t_\n'
            '2-3\t30-37\tanother\tDT\t_\tsame\\_type\tanother\t1-3[2_0]\n'
            '2-4\t38-46\tsentence\t_\t_\t_\t_\t_\n'
            '2-5\t46-47\t.\t_\t_\t_\t_\t_\n'
        )
        self.assertEqual(expected, tsv)


class WebannoTsvFromFile(unittest.TestCase):

    def setUp(self) -> None:
        self.doc = Document.from_file(test_file('test_new.tsv'))

    def test_read_new_format(self):
        self.assertEqual(2, len(self.doc.sentences))
        self.assertEqual(10, len(self.doc.tokens))
        self.assertEqual(3, len(self.doc.layers))
        self.assertEqual(3, len(self.doc.layers['pos']))
        self.assertEqual(2, len(self.doc.layers['lemma']))
        self.assertEqual(1, len(self.doc.layers['relations']))