from text_processor_bm25 import TextProcessorBM25
text = "I want to start working as a programmer. I'm learning Python in Tel-Ran. \
    The teacher says that I have to practice a lot. \
        But I don't have much time for practice. \
            I have to work and take care of my family. \
                I don't know how to find time for practice. "
textProcessor = TextProcessorBM25(text=text)  
print(textProcessor.get_answer  ("Which programming language are you learning in Tel-Ran?") )
textProcessor.save_model("bm25_model.joblib")           