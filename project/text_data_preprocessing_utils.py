import string
import re
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string

def get_stemmer():
    return PorterStemmer()

def get_lemmatizer():
    return WordNetLemmatizer()

def preprocessing(text, stemmer=None, lemmatizer=None):
    if stemmer and lemmatizer:
        raise ValueError("Cannot do stemmatization and lemmatization at the same time")
    
    lower_case_corpus = text.lower()
    tokenized_corpus = word_tokenize(lower_case_corpus, language="english")
    
    if stemmer:
        tokenized_corpus = [stemmer.stem(token) for token in tokenized_corpus]
    elif lemmatizer:
        tokenized_corpus = [lemmatizer.lemmatize(token) for token in tokenized_corpus]

    stops_words_english = set(_get_english_stop_words()) | set(string.punctuation)
    return " ".join([token for token in tokenized_corpus if token not in stops_words_english and _is_alpha_or_hyphenated_word(token)])

def build_vocab(corpus, stemmer=None, lemmatizer=None):
    if stemmer and lemmatizer:
        raise ValueError("Cannot do stemmatization and lemmatization at the same time")
    
    merged_corpus =  " ".join([item_descr.lower() for item_descr in corpus])
    vocab_before_preprocessing = set(word_tokenize(merged_corpus.lower(), language="english"))
    
    if stemmer:
        vocab = [stemmer.stem(token) for token in vocab_before_preprocessing]
    elif lemmatizer:
        vocab = [lemmatizer.lemmatize(token) for token in vocab_before_preprocessing]
    
    stops_words_english = set(_get_english_stop_words() + list(string.punctuation))
    vocab = set([token for token in vocab if token not in stops_words_english and _is_alpha_or_hyphenated_word(token)])
    
    print(f"Vocab size before preprocessing: {len(vocab_before_preprocessing)}\nVocab size after preprocessing: {len(vocab)}")
    return vocab

def _get_english_stop_words():
    stopwords = []
    # source http://members.unine.ch/jacques.savoy/clef/ -> english stop words
    with open("stopwords.txt", "r") as stopwords_dataset_file:
        lines = stopwords_dataset_file.readlines()
        for l in lines:
            # Remove the new line symbol from each word
            stopwords.append(l.removesuffix('\n'))
    return stopwords

def _is_alpha_or_hyphenated_word(token):
    return re.fullmatch(r'[a-zA-Z]+(?:-[a-zA-Z]+)*', token.lower()) is not None
