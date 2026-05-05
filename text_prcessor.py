import joblib
from abc import ABC, abstractmethod 
class TextProcessor(ABC):
    def __init__(self,*,text="",file=""):
        if text and file:
            raise ValueError("Only one of 'text' or 'file' should be provided.")
        elif text:
            self.model = self.create_model(text)
        elif file:
            self.model = self.load_model(file)
        else:
            raise ValueError("Either 'text' or 'file' must be provided.")
    