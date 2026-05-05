import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer  
text = "I want to start working as a programmer. I'm learning Python in Tel-Ran. \
    The teacher says that I have to practice a lot. \
        But I don't have much time for practice. \
            I have to work and take care of my family. \
                I don't know how to find time for practice. "  
text_arr = text.split('.')
cleaned_text_arr = [res for sentence in text_arr if (res := sentence.strip())]
vectorizer = TfidfVectorizer(stop_words="english")
embeddings_text = vectorizer.fit_transform(cleaned_text_arr) #set vectorizer model with text    
query = "Kukureku"
embeddings_query = vectorizer.transform([query]) #get vector for query
sims = cosine_similarity(embeddings_text, embeddings_query) #get cosine similarity between query and text   
sentenceInd = np.argmax(sims) #get index of the most similar sentence
if sims[sentenceInd][0] > 0: #if similarity is greater than 0, print the most similar sentence
    print("Most similar sentence:", cleaned_text_arr[sentenceInd])
else:
    print("No similar sentence found.")