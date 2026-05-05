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
query  = "How much time do you have for practice?"
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(cleaned_text_arr)
query_vector = vectorizer.transform([query])
dense_matrix = tfidf_matrix.toarray()
print(dense_matrix)
similarity_scores = cosine_similarity(query_vector, tfidf_matrix)
print("Similarity scores:", similarity_scores)
most_similar_index = np.argmax(similarity_scores)
most_similar_sentence = cleaned_text_arr[most_similar_index]
print("Most similar sentence:", most_similar_sentence)             
                