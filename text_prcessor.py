import joblib
from abc import ABC, abstractmethod 
class TextProcessor(ABC):
    def __init__(self,*,text="",file=""):
        if text and file:
            raise ValueError("Only one of 'text' or 'file' should be provided.")
        elif text:
           self.create_model(text)
        elif file:
           self.load_model(file)
        else:
            raise ValueError("Either 'text' or 'file' must be provided.")
    @abstractmethod
    def create_model(self,text): pass
    @abstractmethod
    def load_model(self,file): pass
    @abstractmethod
    def save_model(self,file):pass
    @abstractmethod
    def get_answer(self, query): pass
    