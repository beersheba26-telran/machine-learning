import numpy as np
from rank_bm25 import BM25Okapi

text = "I want to start working as a programmer. I'm learning Python in Tel-Ran. \
    The teacher says that I have to practice a lot. \
        But I don't have much time for practice. \
            I have to work and take care of my family. \
                I don't know how to find time for practice. "

text_arr = text.split('.')
cleaned_text_arr = [res for sentence in text_arr if (res := sentence.strip())]

query = "How much time do you have for practice?"

# Tokenize documents and query
tokenized_docs = [doc.lower().split() for doc in cleaned_text_arr]
tokenized_query = query.lower().split()

# Build BM25 model on documents only
bm25 = BM25Okapi(tokenized_docs)

# Get scores for the query against all documents
scores = bm25.get_scores(tokenized_query)
print("BM25 scores:", scores)

most_similar_index = np.argmax(scores)
most_similar_sentence = cleaned_text_arr[most_similar_index]
print("Most similar sentence:", most_similar_sentence)
