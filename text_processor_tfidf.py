from os import getenv

from text_prcessor import TextProcessor
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from logging_config import logger
import joblib
class TextProcessorTFIDF(TextProcessor):
    # constructor should call constructor of TextProcessor class
    def __init__(self,*,text="",file=""):
        super().__init__(text=text, file=file)
    def create_model(self,text):
        logger.debug(f"Creating TFIDF model with text length: {len(text)}")
        text_arr = text.split('.')
        self.cleaned_text_arr = [res for sentence in text_arr if (res := sentence.strip())]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.embeddings_text = self.vectorizer.fit_transform(self.cleaned_text_arr)
    def load_model(self,file):   
        logger.debug(f"Loading TFIDF model from file: {file}")
        data = joblib.load(file)
        self.vectorizer = data['vectorizer']
        self.cleaned_text_arr = data['cleaned_text_arr']
        self.embeddings_text = self.vectorizer.transform(self.cleaned_text_arr)   
    def save_model(self,file):
        logger.debug(f"Saving TFIDF model to file: {file}")
        data = {
            'vectorizer': self.vectorizer,
            'cleaned_text_arr': self.cleaned_text_arr
        }
        joblib.dump(data, file) 
    def get_answer(self, query):  
        logger.debug(f"Getting answer for query: {query}")
        embeddings_query = self.vectorizer.transform([query])
        sims = cosine_similarity(self.embeddings_text, embeddings_query)
        sentenceInd = np.argmax(sims)
        return self.cleaned_text_arr[sentenceInd] if sims[ sentenceInd][0] > 0 else "No similar sentence found."
    @staticmethod
    def createTextProcessor():
        '''Factory method to create a TextProcessorTFIDF instance based on environment variable.
        Expects TFIDF_MODEL_FILE environment variable to be set for loading the model.
        '''
        fileName = getenv("TFIDF_MODEL_FILE")
        if not fileName:
            raise ValueError("TFIDF_MODEL_FILE environment variable is not set.")
        textProcessor = TextProcessorTFIDF(file=fileName)
        return textProcessor