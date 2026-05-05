import joblib
from abc import ABC, abstractmethod 
class TextProcessor(ABC):
    def __init__(self,*,text="",file=""):
        if text and file:
            raise ValueError("Only one of 'text' or 'file' should be provided.")
        elif text:
            self.model = self.create_model(text)
        elif file:
            self.model = joblib.load(file)
        else:
            raise ValueError("Either 'text' or 'file' must be provided.")
    @abstractmethod
    def create_model(self,text): pass
    def save_model(self,file):
        joblib.dump(self.model, file)
    @abstractmethod
    def get_answer(self, query): pass
    