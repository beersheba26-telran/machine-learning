from text_processor_bm25 import TextProcessorBM25
textProcessor = TextProcessorBM25(file="bm25_model.joblib")
print(textProcessor.get_answer("Which programming language are you learning in Tel-Ran?"))