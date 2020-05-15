import string, math, re, io
import pandas as pd
import numpy as np
from sklearn.feature_extraction import stop_words
from nltk.stem import SnowballStemmer



import string, pandas as pd, numpy as np, pickle as pkl

"""
Erick Venegas
04-20-20
hw5.py
Sean Current

Program measures how similiar documents are to one another.
Used in practice to categorize documents by subject or style, or even attempt to determin who wrote a text.
"""

def dot_product(vector_a, vector_b):
    """
    Function takes in two word vectors in form of dictionaries, and returns their dot product.
    If two dictionaries have the same key, multiply their values, and add to final count.
    Do this for every key in dictionary.
    
    PARAMETERS: vector_a -- a dictionary in the form {'dot': 3, 'string': 2}
                vector_b -- a dictionary in the form {'dot': 1, 'string': 3}
    
    RETURNS: an int or float, the dot product between both vector_a and vector_b
    """
    dot_product = 0

    for k in vector_a:
        dot_product += vector_a[k] * vector_b.get(k, 0)

    return dot_product

def magnitude(vector):
    """
    Function takes in a single word vector (in the form of a dictionary) and returns it's magnitude.
    By itterating dictionary and multiplying each value to the 2nd power, summing new values, and calculating
    the square root of the summed value.

    PARAMETERS: vector -- a dictionary in the form {'a' : 1, 'b' : 2, 'c' : 3}
    
    RETURNS: magnitude of word vector
    """
    magnitude = 0

    for k, v in vector.items():
        magnitude += v**2
    
    return math.sqrt(magnitude)

def cosine_similarity(u, v):
    return dot_product(u, v) / (magnitude(u) * magnitude(v))

def get_text(filename):
    """
    Function takes in a filename of a text file, and returns a single string containing the cleaned-up contents of file.

    PARAMETERS: filename -- a string, the string of the textfile we'll be cleaning

    RETURNS: clean_string -- a string, the file contents of filename parameter.
    With punctuation and digits removed, and made into lowercase.
    """
    with open(filename) as file:
        text = file.read()
        lowered_text = text.replace("n't", '')

        for char in lowered_text:
            if char in string.punctuation or char in string.digits:
                lowered_text = lowered_text.replace(char, "")

        return lowered_text.lower()

def vectorize(filename, stop_words, stemmer):
    """
    Function takes in a filename, stopwards list, and stemmer and return dictionary representing word count vector mapping
    cleaned words from the file to wordcounts.
    Checks to see if word is in dictionary, if so, increment. Otherwise, set word count to 1.

    PARAMETERS: filename -- a string, the name of the file that will be read in
                stop_words -- a string of strings
                stemmer -- SnowballStemmer object used to check language
    
    RETURNS: d -- a dictionary mapping a word to the number of times it appears
    """
    clean = get_text(filename)
    lowered_text = clean.split()
    d = {}
    for token in lowered_text:
        token = stemmer.stem(token)
        if token not in stop_words:
            if token in d:
                d[token] += 1
            else:
                d[token] = 1
    return d


def get_doc_freqs(word_vectors):
    """
    Function takes in a list of wordcount vectors (in form of dictionaries) and returns a dictionary that maps
    each key from all of the vectors to the number of vectors that word appears in
    Checks to see if word is in dictionary, if so, increment. Otherwise, set word count to 1.

    PARAMETERS: word_vectors -- a list of dictionary vectors where a word maps to it's appearance count

    RETURNS: new -- a dictionary mapping a word to the number of times it appears
    """
    new = {}
    for d in word_vectors:
        for k, v in d.items():
            if k in new:
                new[k] += 1
            else:
                new[k] = 1
    return new

def tfidf(word_vectors):
    """
    Function takes in a list of wordcount vectors (in form of dictionaries) and replaces the word counts
    (values of dictionary) with TF-IDF measurements.

    PARAMETERS: word_vectors -- a list of dictionary vectors where a word maps to it's appearance count

    RETURNS: N/A
    """
    all_freqs = get_doc_freqs(word_vectors)

    if len(word_vectors) >= 100:
        scale = 1
    else:
        scale = 100/len(word_vectors)

    for key in word_vectors:
        for item in key:
            key[item] = key[item] * (1 + math.log2(scale * (len(word_vectors)/all_freqs[item])))

def get_similarity_matrix(filenames, stop_words, stemmer):
    """
    this function takes a list of filenames, a stopwords list, and a stemmer, 
    and returns a DataFrame containing the matrix of document similarities.
    Your DataFrame should use the filenames for both its index and its column names.

    PARAMETERS: filenames -- a list of strings, which represent files that will be read from
                stop_words -- a list of stings, which will be used as our stop words
                stemmer -- SnowballStemmer object used for 'english'

    RETURNS: df -- a dataframe, with cosine similarity between the TF-IDF vectors for each document/file as values.
    """
    df = pd.DataFrame(index=filenames, columns=filenames)
    l = [vectorize(file, stop_words, stemmer) for file in filenames]

    tfidf(l)

    for i in range(len(df)):
        j = i+1
        while j < len(df.iloc[i]):
            df.iloc[i, j] = cosine_similarity(l[i], l[j])
            df.iloc[j, i] = cosine_similarity(l[i], l[j])
            j+=1
        df.iloc[i, i] = 1

    return df     

def main():
    """
    Constructs a similarity matrix from every text file starting with '0000'

    PARAMETERS: N/A

    RETURNS: N/A
    """
    l = ['0000{}.txt'.format(str(i)) for i in range(6)]
    stops = list(stop_words.ENGLISH_STOP_WORDS) + ['did', 'gone', 'ca']
    stemmer = SnowballStemmer("english")
    df = get_similarity_matrix(l, stops, stemmer)
    print(df)

main()

