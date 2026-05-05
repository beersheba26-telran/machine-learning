from text_processor_tfidf import TextProcessorTFIDF
textProcessor = TextProcessorTFIDF(file="tfidf_model.joblib")
print(textProcessor.get_answer("Which programming language are you learning in Tel-Ran?"))