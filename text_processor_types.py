from text_processor_tfidf import TextProcessorTFIDF
from text_processor_bm25 import TextProcessorBM25
processorTypes = {
    "TFIDF": TextProcessorTFIDF.createTextProcessor,
    "BM25": TextProcessorBM25.createTextProcessor
}