from text_processor_tfidf import TextProcessorTFIDF
text = "I want to start working as a programmer. I'm learning Python in Tel-Ran. \
    The teacher says that I have to practice a lot. \
        But I don't have much time for practice. \
            I have to work and take care of my family. \
                I don't know how to find time for practice. "
textProcessor = TextProcessorTFIDF(text=text)  
print(textProcessor.get_answer  ("Whcich programming language are you learning in Tel-Ran") )
textProcessor.save_model("tfidf_model.joblib")           