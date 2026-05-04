import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer  
text = "I want to start working as a programmer. I'm learning Python in Tel-Ran. \
    The teacher says that I have to practice a lot. \
        But I don't have much time for practice. \
            I have to work and take care of my family. \
                I don't know how to find time for practice. "  
    
                