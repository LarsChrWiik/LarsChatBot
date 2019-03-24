
import pandas as pd
import nltk
#nltk.download('stopwords')  
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize.toktok import ToktokTokenizer

noise = pd.read_csv('noise_words.txt', sep=";", encoding='latin1', header=None).iloc[:, 0].values.tolist()
tokenizer = ToktokTokenizer()
stop_words = set(stopwords.words('norwegian'))
stemmer = SnowballStemmer("norwegian")


def convert_to_bag_of_words(sentence, verbose=False):

    # Tokenization
    # words = nltk.word_tokenize(question)
    words = tokenizer.tokenize(sentence)
    if verbose: print("Tokenize:", words)

    # Lower case.
    words = [x.lower() for x in words]
    if verbose: print("lowercase:", words)

    # Remove noise tokens.
    words = [x for x in words if x not in noise]
    if verbose: print("remove noise tokens:", words)

    # Remove noise within tokens.
    for i, w in enumerate(words):
        for n in noise:
            words[i] = words[i].replace(n, '')
    if verbose: print("remove noise in tokens:", words)

    # Remove stop words
    words = [x for x in words if x not in stop_words]
    if verbose: print("remove stop words:", words)

    # Stemming
    words = [stemmer.stem(x) for x in words]
    if verbose: print("Stemming:", words)

    return words
