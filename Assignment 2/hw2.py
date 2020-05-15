"""
Erick Venegas
ISTA331
02-20-20
hw2.py
Collab: Sami Robbins
"""

from sklearn.feature_extraction import stop_words
from nltk.stem import SnowballStemmer
from sklearn.metrics import confusion_matrix
import math, string, random, os

class NaiveBayesClassifier:
    def __init__(self, labeled_data, k=0.5,  max_words=50):
        self.labeled_data = labeled_data
        self.k = k
        self.max_words = max_words
        self.stemmer = SnowballStemmer('english')
        self.spam = 0
        self.ham = 0
        self.word_probs = self.count_words()

        for num in self.labeled_data.y:
            if num == 0:
                self.ham += 1
            else:
                self.spam += 1

        for w in self.word_probs:
            self.word_probs[w][0] = (self.word_probs[w][0] + k)/(self.spam + 2 * k)
            self.word_probs[w][1] = (self.word_probs[w][1] + k)/(self.ham + 2 * k)

    def tokenize(self, text):
        lowered_text = text.lower().replace("n't", '')
        valid_tokens = set()

        for char in lowered_text:
            if char in string.punctuation or char in string.digits:
                lowered_text = lowered_text.replace(char, "")

        for token in lowered_text.split():
            token = self.stemmer.stem(token)
            if token not in stop_words.ENGLISH_STOP_WORDS:
                valid_tokens.add(token)
        
        return valid_tokens

    def count_words(self):
        d = {}
        for i in range(len(self.labeled_data.X)):
            token_data = self.tokenize(self.labeled_data.X[i])

            for element in token_data:
                temp = (self.labeled_data.y[i])
                if element not in d:
                    d[element] = [0, 0]
                if temp:
                    d[element][0] += 1
                else:
                    d[element][1] += 1
                 
        return d

    def spam_probability(self, email):
        pass
        #returns float of spammy calculation

    def get_tokens(self, text):
        '''
        Takes a feature vector (the tokenized text) and returns a random sample of tokens from the vector
        the number or tokens should be equal to max_word, or the length of the vector if it is less than max_word
        '''
        k = min(self.max_words, len(text))
        return random.sample(sorted(text), k)

    def classify(self, fv):
        '''
        takes a feture vector
        classifies the feature vector as fiction or nonfiction using fiction_prob
        fiction/nonfiction scale ranges from 0 (nonfiction) to 1 (fiction) with .5 being the median value on the scale
        returns True if fiction and False if nonfiction
        '''
        return self.spam_probability(fv) >= 0.5
        
    def predict(self, X):
        '''
        note that this function takes an argument X
        although we will be using self to classify the feature vectors, the vectors we are classifying are ones that our classifier has not seen
        '''
        predictions = []
        for fv in X:
            predictions.append(self.classify(fv))
        return predictions

class LabeledData:
    """
    Instance takes two strings which are the paths to files (default arguments of 'data/ 2002/easy_ham' and 'data/2002/spam').
    If first parameter is None; traverse the filenames in those directories, pass each filename to parse_message,
    and append the parsed email to a list(type str). 

    PARAMETERS: x -- a string, the path of the directory to fetch valid email files
                y -- a string, the path of the directory to fetch spam email files
                data_matrix -- a matrix
                labels -- a list vector, containing 0's where the corresponding emails are ham and 1's where they are spam.
    RETURNS: None

    """
    def __init__(self, x='data/2002/easy_ham', y='data/2002/spam', data_matrix=None, labels=None):
        if data_matrix is None:
            self.X = []
            self.y = []

            for filename in sorted(os.listdir(x)):
                self.X.append(self.parse_message(x + "/" + filename))
                self.y.append(0)
            
            for filename in sorted(os.listdir(y)):
                self.X.append(self.parse_message(y + "/" + filename))
                self.y.append(1)
        else:
            self.X = X
            self.y = y

    def parse_message(self, filename):
        """
        Itterates through every file in 'data/2002/easy_ham/ directory, for every file, read file contents.
        Returns a string containing a cleaned version of the email

        PARAMETERS: filename -- a string, name of the file to extract contents

        RETURNS: a cleaned, valid string to add to builder
        """
        with open(filename, errors='ignore', encoding="ascii") as file:
            result = []
            line = file.readline().strip()

            while line is not None:
                if line.startswith("Subject:"):
                    remove = True
                    for token in line[8:].split():
                        if token.lower() == "re:" and remove:
                            continue
                        else:
                            result.append(token)
                            remove = False
                line = file.readline().strip()
            for line in file:
                res = LabeledData.parse_line(line)
                if res:
                    result.append(res)

        final_string = " ".join(result)
        return final_string

    @staticmethod
    def parse_line(line):
        """
        method takes in line from an email and returns stripped line if not a header. Otherwise, return the empty string

        PARAMETERS: line -- a string,

        RETURNS: a concat string to add to builder
        """
        new_line = line.strip()

        if ':' in new_line:
            colon = new_line.find(':')
            lis = new_line[:colon].split()

            if len(lis) == 1:
                return ""

        return new_line

def main():
    # pass
    # training = LabeledData()
    # testing = LabeledData('data/2003/easy_ham', 'data/2003/spam')
    # nbc = NaiveBayesClassifier(training, 25)

    training_data = LabeledData()
    testing_data = LabeledData('data/2003/easy_ham', 'data/2003/spam')
    nbc = NaiveBayesClassifier(training_data, 25)
    nbc_predict = nbc.predict(testing_data.X)
    c_matrix = confusion_matrix(testing_data.y, nbc_predict)
    dia_sum = sum(c_matrix.diagonal())
    final_result = (dia_sum/c_matrix.sum()) * 100

    print(c_matrix)
    print("accuracy: {}%".format(str(round(final_result, 2))))

if __name__ == "__main__":
    main()
    
    
    