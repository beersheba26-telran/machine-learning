import joblib
from rank_bm25 import BM25Okapi
import re
text = "I want to start working as a programmer. I'm learning Python in Tel-Ran. \
    The teacher says that I have to practice a lot. \
        But I don't have much time for practice. \
            I have to work and take care of my family. \
                I don't know how to find time for practice. "  
def clean_text(sentence:str):
   sentence = sentence.lower()
   sentence = re.sub(r'[^a-z]', ' ', sentence).strip()
   return sentence
text_arr = text.split('.')
cleaned_text_arr = [res for sentence in text_arr if (res := clean_text(sentence))]
words = [doc.split() for doc in cleaned_text_arr]
bm25 = BM25Okapi(words)
# saving model means saving the text_arr and the bm25 model
query = "Which programming language are you learning in Tel-Ran?"
cleaned_query = clean_text(query)
query_words = cleaned_query.split()
scores = bm25.get_scores(query_words)
best_doc_index = scores.argmax()
print(text_arr[best_doc_index])