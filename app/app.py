#import library
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from spacy import displacy



Class App():
    """
    This class is for natural language processing tasks.
    It provides methods for sentiment analysis, entity extraction,
     stopword removal and text preprocessing.
    """

    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words("english"))
        self.nlp = spacy.load("en_core_web_sm")


    def get_sentiment(self, text):
        # Uses SentimentIntensityAnalyzer to calculate sentiment of input text.

        # Args: text(str) Returns: A dictionary containing sentiment scores
        return self.sia.polarity_scores(text)


    def get_entities(self, text):
        # Uses en_core_web_sm spaCy model to extract entities from input
        # returns a list of tuples containing the entity text and label.
        doc = self.nlp(text)
        return [(ent.text, ent.label_) for ent in doc.ents]


    def get_tokens(self, text):
        # Uses NLTKs word_tokenize to split input into individual tokens
        return word_tokenize(text)


    def remove_stopwords(self, tokens):
        # filters stopwords from tokenized list and removes them
        return [word for word in tokens if word.lower() not in self.stop_words]


    def process_text(self, text):
        # combines tokenization and stopword removal to preprocess input
        tokens = [word.lower() for word in self.get_tokens(text) if word.lower() not in self.stop_words]
        return " ".join(tokens)


    def visualize_entities(self, text):
        # uses spacys displacy to visualize extracted entities in input
        # Args text(str) Returns: str: The HTML code for the visualization
        if not text:
            raise ValueError("Input text cannot be empty")
        doc = self.nlp(text)
        return displacy.render(doc, style="ent", jupyter=True)

