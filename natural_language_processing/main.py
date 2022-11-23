import numpy as np
import spacy
from spacy import displacy
from spacy.matcher import Matcher

# Spacy Objects
# doc = document / list of all tokens
# sent = sentence
# token = token
# span = phrase / one or more tokens

# Preprocess
# .replace("-\n", "").replace("\n", " ")


class Spacy:
    """
    Source
    Video: https://youtu.be/dIUTsFT2MeQ
    Textbook: https://spacy.pythonhumanities.com/intro.html
    """

    def __init__(self):
        with open("data/farm.txt") as f:
            self.text = f.read()
        # print(text)

        # Build upon small model
        # python -m spacy download en_core_web_sm
        self.sm_nlp = spacy.load("en_core_web_sm")

        # Build upon ??? model
        # python -m spacy download en_core_web_md
        self.md_nlp = spacy.load("en_core_web_md")

        # Build upon blank model
        self.blank_nlp = spacy.blank("en")

    @staticmethod
    def word_tokenize(doc):
        for token in doc:
            print(token)

    @staticmethod
    def sent_tokenize(doc):
        # Sentence boundary detection
        for sentence in doc.sents:
            print(sentence)

    # Section 1: Building Blocks of spaCy 3
    def linguistic_annotation(self):
        # Create Doc container
        doc = self.sm_nlp(self.text)

        # Tokenization
        self.word_tokenize(doc)
        self.sent_tokenize(doc)

        # Token attribute
        token = doc[1]
        print(
            token,
            token.text,
            token.head,
            token.left_edge,
            token.right_edge,
            token.ent_type,
            token.ent_type_,
            token.ent_iob_,
            token.lemma_,
            token.morph,
            token.pos_,
            token.dep_,
            token.lang_,
            sep="\n",
        )

        # Part of Speech Tagging (POS)
        sentence1 = list(doc.sents)[0]
        for token in sentence1:
            print(token.text, token.pos_, token.dep_)
        displacy.render(sentence1, style="dep")  # work in Jupyter

        # Named Entity Recognition
        for ent in doc.ents:
            print(ent.text, ent.label_)
        displacy.render(doc, style="ent")  # work in Jupyter

    def word_vector(self):
        # Similar words using word vectors
        word = "pig"
        ms = self.md_nlp.vocab.vectors.most_similar(
            np.asarray([self.md_nlp.vocab.vectors[self.md_nlp.vocab.strings[word]]]),
            n=10,
        )
        words = [self.md_nlp.vocab.strings[w] for w in ms[0][0]]
        distances = ms[2]
        print(words)
        print(distances)

        # Document/Word similarity
        doc1 = self.md_nlp("I like pig.")
        doc2 = self.md_nlp("I like duck.")
        print(doc1)
        print(doc2)
        print(doc1.similarity(doc2))

        word1 = self.md_nlp("pig")
        word2 = self.md_nlp("milker")
        print(word1)
        print(word2)
        print(word1.similarity(word2))

    def pipeline(self):
        # Add pipe
        self.blank_nlp.add_pipe("sentencizer")
        doc = self.blank_nlp(self.text)
        print(list(doc.sents))
        print(len(list(doc.sents)))

        # Analyse pipe
        print(
            self.blank_nlp.analyze_pipes(),
            self.md_nlp.analyze_pipes(),
            self.sm_nlp.analyze_pipes(),
            sep="\n",
        )

    # Section 2: Rules-based spaCy
    def entity_ruler(self):
        text = "West Chestertenfieldville was referenced in Mr. Deeds."
        doc = self.sm_nlp(text)
        for ent in doc.ents:
            print(ent.text, ent.label_)

        entity_ruler = self.sm_nlp.add_pipe("entity_ruler", before="ner")
        patterns = [
            {"label": "GPE", "pattern": "West Chestertenfieldville"},
            {"label": "FILM", "pattern": "Mr. Deeds"},
        ]
        entity_ruler.add_patterns(patterns)
        print(self.sm_nlp.analyze_pipes())

        doc = self.sm_nlp(text)
        for ent in doc.ents:
            print(ent.text, ent.label_)

    def matcher(self):
        # Build NLP model -> matcher -> doc -> result/matches
        matcher = Matcher(self.sm_nlp.vocab)
        pattern = [
            {"LIKE_EMAIL": True}
        ]  # [{"POS": "PROPN", "OP": "+"}, {"POS": "VERB"}]
        matcher.add("EMAIL_ADDRESS", [pattern], greedy="LONGEST")

        text = "This is an email address: wmattingly@aol.com"
        doc = self.sm_nlp(text)
        matches = matcher(doc)
        matches.sort(key=lambda x: x[1])
        print(matches)  # format: lexeme, start token, end token
        for match in matches:
            print(
                match,
                self.sm_nlp.vocab[match[0]].text,
                doc[match[1] : match[2]],
            )


if __name__ == "__main__":
    spacy_object = Spacy()

    # Section 1: Building Blocks of spaCy 3
    spacy_object.linguistic_annotation()
    spacy_object.word_vector()
    spacy_object.pipeline()

    # Section 2: Rules-based spaCy
    spacy_object.entity_ruler()
    spacy_object.matcher()

    # https://youtu.be/dIUTsFT2MeQ?t=5762
    # http://spacy.pythonhumanities.com/02_01_entityruler.html
    nlp = spacy.load("en_core_web_sm")
    doc = nlp("The little black dog barked at the white cat and chased away")
    sentence1 = list(doc.sents)[0]
    displacy.render(sentence1, style="dep")
