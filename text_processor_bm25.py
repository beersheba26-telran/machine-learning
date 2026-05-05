from os import getenv

from text_prcessor import TextProcessor
import joblib
from rank_bm25 import BM25Okapi
class TextProcessorBM25(TextProcessor):
    def __init__(self,*,text="",file=""):
        super().__init__(text=text, file=file)
    def create_model(self,text):
        text_arr = text.split('.')
        self.cleaned_text_arr = [res for sentence in text_arr if (res := sentence.strip())]
        tokenized_corpus = [doc.split() for doc in self.cleaned_text_arr]
        self.bm25 = BM25Okapi(tokenized_corpus)
    def load_model(self,file):   
        data = joblib.load(file)
        self.cleaned_text_arr = data['cleaned_text_arr']
        self.bm25 = data['bm25']
    def save_model(self,file):
        data = {
            'cleaned_text_arr': self.cleaned_text_arr,
            'bm25': self.bm25
        }
        joblib.dump(data, file) 
    def get_answer(self, query):  
        tokenized_query = query.split(" ")
        doc_scores = self.bm25.get_scores(tokenized_query)
        sentenceInd = doc_scores.argmax()
        return self.cleaned_text_arr[sentenceInd] if doc_scores[sentenceInd] > 0 else "No similar sentence found."
    @staticmethod
    def createTextProcessor():
        '''Factory method to create a TextProcessorBM25 instance based on environment variable.
        Expects BM25_MODEL_FILE environment variable to be set for loading the model.
        '''
        fileName = getenv("BM25_MODEL_FILE")
        if not fileName:
            raise ValueError("BM25_MODEL_FILE environment variable is not set.")
        textProcessor = TextProcessorBM25(file=fileName)
        return textProcessor