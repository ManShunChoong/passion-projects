import re
import string

import matplotlib.pyplot as plt
import nltk
import numpy as np
import spacy
from graphviz import Digraph
from nltk.corpus import wordnet
from spacy import displacy
from spacy.matcher import Matcher
from textblob import TextBlob
from wordcloud import WordCloud

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


# Tokenization
def sent_tokenize():
    # nltk.download('punkt')

    text = "Hello duck, I'm pig. How are you?"

    # RegEx: Split by period, exclamation mark and question mark
    sentences = re.split(r"[.!?]", text)
    print(sentences)

    # NLTK
    sentences = nltk.tokenize.sent_tokenize(text=text)
    print(sentences)


def word_tokenize():
    # nltk.download('punkt')

    text = "Hello duck, I'm pig. How are you?"
    words = text.split()
    print(words)

    # RegEx: Split by space
    words = re.split(r" ", text)
    print(words)

    # RegEx: Split by number(s)
    words = re.split(r"\d+", text)
    print(words)

    # RegEx: Split by some special characters
    words = re.split(r"[-+#]", text)
    print(words)

    # NLTK
    words = nltk.tokenize.word_tokenize(text=text)
    print(words)

    tokens_text = nltk.Text(words)
    print(tokens_text[0 : len(tokens_text)])

    # TextBlob
    blob = TextBlob(text)
    print(blob.words)


# Stemming
def stem():
    token = "cries"
    suffixes = ["ing", "ly", "ed", "ious", "ies", "ive", "es", "s", "ment"]
    for suffix in suffixes:
        if token.endswith(suffix):
            print(token[: -len(suffix)])
            break

    # RegEx: Get suffix
    suffix_pattern = (
        r"^.*(ing|ly|ed|ious|ies|ive|es|s|ment)$"
        or r"(ing|ly|ed|ious|ies|ive|es|s|ment)$"
    )
    print(re.findall(suffix_pattern, "processing"))
    print(re.findall(suffix_pattern, "ing"))
    print(re.findall(suffix_pattern, "hello"))

    # RegEx: Get word with suffix
    suffix_word_pattern = (
        r"^.*(?:ing|ly|ed|ious|ies|ive|es|s|ment)$"
        or r".*(?:ing|ly|ed|ious|ies|ive|es|s|ment)$"
    )
    print(re.findall(suffix_word_pattern, "processing"))
    print(re.findall(suffix_word_pattern, "ing"))
    print(re.findall(suffix_word_pattern, "hello"))

    # RegEx: Split word and its suffix
    split_word_pattern = (
        r"^(.*)(ing|ly|ed|ious|ies|ive|es|s|ment)$"
        or r"(.*)(ing|ly|ed|ious|ies|ive|es|s|ment)$"
    )
    print(re.findall(split_word_pattern, "processing"))
    print(re.findall(split_word_pattern, "ing"))
    print(re.findall(split_word_pattern, "hello"))

    # RegEx: Stem word
    stem_word_pattern = (
        r"^.*(?=ing|ly|ed|ious|ies|ive|es|s|ment|)$"
        or r"^.*(?=ing|ly|ed|ious|ies|ive|es|s|ment|)"
    )
    print(re.findall(stem_word_pattern, "processing"))
    print(re.findall(stem_word_pattern, "ing"))
    print(re.findall(stem_word_pattern, "hello"))

    # NLTK
    print(nltk.PorterStemmer().stem(token))
    print(nltk.LancasterStemmer().stem(token))
    print(nltk.SnowballStemmer(language="english").stem(token))


# Lemmatization
def lemmatize():
    # nltk.download('wordnet')

    # NLTK
    wnl = nltk.WordNetLemmatizer()
    print(wnl.lemmatize("rocks"))
    print(wnl.lemmatize("produced", pos="v"))
    print(wnl.lemmatize("better", pos="a"))
    print(wnl.lemmatize("women", pos="n"))

    tokens = ["List", "listed", "lists", "listing", "listings"]
    for token in tokens:
        print(wnl.lemmatize(token))
    for token in tokens:
        print(wnl.lemmatize(token, pos="v"))

    # TextBlob
    blob = TextBlob(" ".join(tokens))
    tokens = blob.words
    print(tokens.lemmatize())

    text = (
        "DENNIS: Listen, strange women lying in ponds distributing swords is no basis for a system of government.  "
        "Supreme executive power derives from a mandate from the masses, not from some farcical aquatic ceremony. "
    )
    blob = TextBlob(text)
    tokens = blob.words
    print(tokens.lemmatize())


# Lemmatization with POS Tags Specifications
def lemmatize_pos():
    def get_wordnet_pos(word):
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {
            "J": wordnet.ADJ,
            "N": wordnet.NOUN,
            "V": wordnet.VERB,
            "R": wordnet.ADV,
        }
        return tag_dict.get(tag, wordnet.NOUN)

    lemmatizer = nltk.WordNetLemmatizer()

    word = "feet"
    print(lemmatizer.lemmatize(word, get_wordnet_pos(word)))

    sentence = "The striped bats are hanging on their feet for best"
    print(
        [
            lemmatizer.lemmatize(w, get_wordnet_pos(w))
            for w in nltk.word_tokenize(sentence)
        ]
    )


# Indexing
def index():
    rotokas_words = nltk.corpus.toolbox.words("rotokas.dic")
    cv_word_pairs = [
        (cv, w) for w in rotokas_words for cv in re.findall(r"[ptksvr][aeiou]", w)
    ]
    cv_index = nltk.Index(cv_word_pairs)
    print(cv_index["su"])
    print(cv_index["po"])


# Frequency distribution / Bag of words (EDA)
def freq_distribution():
    # nltk.download('punkt')
    # nltk.download("treebank")

    # Example 1
    words = nltk.corpus.treebank.words()
    strings = [vs for word in words for vs in re.findall(r"[aeiou]{2,}", word)]
    freq_dist = nltk.FreqDist(strings)
    print(freq_dist.most_common())
    print(freq_dist.most_common(10))
    print(freq_dist.items())

    # Example 2
    text = "John likes to watch movies. Mary likes movies too"
    tokens = nltk.tokenize.word_tokenize(text)

    punctuations = list(string.punctuation)
    filtered_tokens = [token for token in tokens if token not in punctuations]

    freq_dist = nltk.FreqDist(filtered_tokens)
    print(freq_dist.most_common())


# Conditional frequency distribution (EDA)
def cond_freq_distribution():
    nltk.download("toolbox")

    rotokas_words = nltk.corpus.toolbox.words("rotokas.dic")

    cond_sample = [
        (cv[0], cv[1])  # (condition, value)
        for w in rotokas_words
        for cv in re.findall(r"[ptksvr][aeiou]", w)
    ]
    cfd = nltk.ConditionalFreqDist(cond_sample)
    cfd.tabulate()


# RegEx: Start with vowel(s) or End with vowel(s) or Contain non-vowel
def regex_example():
    # nltk.download("udhr")

    word = "supercalifragilisticexpialidocious"
    # RegEx: Start with vowel(s) or End with vowel(s) or Non-vowels
    regexp = r"^[AEIOUaeiou]+|[AEIOUaeiou]+$|[^AEIOUaeiou]"
    print("".join(re.findall(regexp, word)))

    english_udhr = nltk.corpus.udhr.words("English-Latin1")
    print(english_udhr)

    tokens = ("".join(re.findall(regexp, token)) for token in english_udhr[:75])
    string = nltk.tokenwrap(tokens)
    print(string)


# Remove stop words and punctuation
def remove_stop_words():
    # nltk.download("stopwords")

    text = (
        "This programme is designed to provide students with knowledge and applied skills in data science, "
        "big data analytics and business intelligence. It aims to develop analytical and investigative knowledge "
        "and skills using data science tools and techniques, and to enhance data science knowledge and critical "
        "interpretation skills. Students will understand the impact of data science upon modern processes and "
        "businesses, be able to identify, and implement specific tools, practices, features and techniques to "
        "enhance the analysis of data."
    )
    word_tokens = nltk.word_tokenize(text)

    stop_words = nltk.corpus.stopwords.words("english")
    punctuations = list(string.punctuation)
    # print(stop_words)
    # print(punctuations)

    filtered_tokens = [
        token
        for token in word_tokens
        if token not in stop_words and token not in punctuations
    ]
    print(filtered_tokens)


# Create word cloud
def create_word_cloud():
    text = (
        "This programme is designed to provide students with knowledge and applied skills in data science, "
        "big data analytics and business intelligence. It aims to develop analytical and investigative knowledge "
        "and skills using data science tools and techniques, and to enhance data science knowledge and critical "
        "interpretation skills. Students will understand the impact of data science upon modern processes and "
        "businesses, be able to identify, and implement specific tools, practices, features and techniques to "
        "enhance the analysis of data."
    )
    word_cloud = WordCloud(background_color="white").generate(text)
    plt.figure(figsize=(15, 8), facecolor=None)
    plt.imshow(word_cloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()


# Segmentation
def segmentation():
    # nltk.download('words')

    text = "thesistermisslow"
    corpus_words = set(
        word.lower() for word in nltk.corpus.words.words() if len(word) > 1
    )

    def f(text, chain=None):
        if chain is None:
            chain = []

        return (
            (
                token
                for chain in (
                    f(text[len(word) :], chain + [word])
                    for word in corpus_words
                    if text.startswith(word)
                )
                for token in chain
            )
            if text
            else {tuple(chain)}
        )

    for tokens in f(text):
        print(" ".join(tokens))


# Part of speech (POS) tagging
def pos_tagging():
    # nltk.download('averaged_perceptron_tagger')

    # NTLK
    text = "And now for something completely different"
    tokens = nltk.word_tokenize(text)
    print(nltk.pos_tag(tokens))

    text = "They refuse to permit us to obtain the refuse permit"
    tokens = nltk.word_tokenize(text)
    print(nltk.pos_tag(tokens))

    text = "I couldn’t get back to sleep"
    tokens = nltk.word_tokenize(text)
    print(nltk.pos_tag(tokens))

    # TextBlob
    text = (
        "Python is a high-level, general-purpose programming language. Python is a high-level, general-purpose "
        "programming language. "
    )
    blob = TextBlob(text)
    print(blob.tags)

    # RegEx
    patterns = [
        (r".*ing$", "VBG"),  # gerunds
        (r".*ed$", "VBD"),  # simple past
        (r".*es$", "VBZ"),  # 3rd singular present
        (r".*ould$", "MD"),  # modals
        (r".*\'s$", "NN$"),  # possessive nouns
        (r".*s$", "NNS"),  # plural nouns
        (r"^-?[0-9]+(.[0-9]+)?$", "CD"),  # cardinal numbers
        (r".*", "NN"),  # nouns (default)
        (r"^\d+$", "CD"),
        (r".*ing$", "VBG"),  # gerunds, i.e. wondering
        (r".*ment$", "NN"),  # i.e. wonderment
        (r".*ful$", "JJ"),  # i.e. wonderful
    ]
    regexp_tagger = nltk.RegexpTagger(patterns)
    tagger = nltk.tag.sequential.RegexpTagger(patterns)

    text1 = nltk.tokenize.word_tokenize(
        "Python is a high-level, general-purpose programming language"
    )
    print(tagger.tag(text1))
    print()
    print(nltk.pos_tag(text1))

    # NLTK: Get meaning of tag
    # nltk.download('tagsets')
    nltk.help.upenn_tagset("JJ")

    # NLTK: Tagged corpora
    print(nltk.corpus.treebank.tagged_words())


# String to tagged tokens
def string_to_tagged_tokens():
    tagged_token = nltk.tag.str2tuple("fly/NN")
    print(tagged_token)

    text = (
        "The/AT grand/JJ jury/NN commented/VBD on/IN a/AT number/NN of/IN "
        "other/AP topics/NNS ,/, AMONG/IN them/PPO the/AT Atlanta/NP and/CC "
        "Fulton/NP-tl County/NN-tl purchasing/VBG departments/NNS which/WDT it/PPS "
        "said/VBD ``/`` ARE/BER well/QL operated/VBN and/CC follow/VB generally/RB "
        "accepted/VBN practices/NNS which/WDT inure/VB to/IN the/AT best/JJT "
        "interest/NN of/IN both/ABX governments/NNS ''/'' ./."
    )
    for token in text.split():
        print(nltk.tag.str2tuple(token))


# Similar words
def similar_words():
    # nltk.download('brown')

    text = nltk.Text(word.lower() for word in nltk.corpus.brown.words())
    words = ["woman", "bought", "over", "the"]
    for word in words:
        print(word)
        text.similar(word)
        print()


# Universal POS tagset
def pos_tagset():
    # nltk.download('universal_tagset')

    tagged_news_tokens = nltk.corpus.brown.tagged_words(
        categories="news", tagset="universal"
    )
    tag_freq_dist = nltk.FreqDist(tag for word, tag in tagged_news_tokens)
    print(tag_freq_dist.most_common())


# Parse tree
def parse_tree():
    # Example 1
    tokens = nltk.tokenize.word_tokenize("I write a book")

    inputs = [
        "S -> NP VP",
        "PP -> P NP",
        "NP -> Det N | PP NP | Det N PP | 'I'",
        "VP -> V NP | VP PP | V",
        "Det -> 'a'",
        "N -> 'book'",
        "V -> 'write'",
    ]
    grammar = nltk.CFG.fromstring(inputs)

    parser = nltk.ChartParser(grammar=grammar)
    for tree in parser.parse(tokens):
        tree.draw()

    # Example 2
    grammar = nltk.CFG.fromstring(
        """
    S -> NP VP 
    PP -> P NP 
    NP -> Det N | Det N PP | 'I' 
    VP -> V NP | VP PP 
    Det -> 'an' | 'my' 
    N -> 'elephant' | 'pajamas' 
    V -> 'shot' 
    P -> 'in' 
    """
    )

    sent = ["I", "shot", "an", "elephant", "in", "my", "pajamas"]
    parser = nltk.ChartParser(grammar)
    for tree in parser.parse(sent):
        tree.draw()


# FSA
def create_fsa():
    fsa = Digraph(name="FSA")
    fsa.attr(rankdir="LR")
    fsa.node("A", "q0", shape="circle")
    fsa.node("B", "q1", shape="circle")
    fsa.node("L", "q2", shape="doublecircle")

    fsa.edge("A", "A", label="b")
    fsa.edge("A", "B", label="a")
    fsa.edge("B", "A", label="b")
    fsa.edge("B", "L", label="a")
    fsa.edge("L", "A", label="b")
    fsa.edge("L", "L", label="a")

    print(fsa.source)
    fsa.view(directory="graphs", filename="fsa")


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

    # sent_tokenize()
    # word_tokenize()
    # stem()
    # lemmatize()
    # lemmatize_pos()
    # index()
    # freq_distribution()
    # cond_freq_distribution()
    # regex_example()
    # remove_stop_words()
    # create_word_cloud()
    # segmentation()
    # pos_tagging()
    # similar_words()
    # string_to_tagged_tokens()
    # pos_tagset()
    # parse_tree()
    # create_fsa()
