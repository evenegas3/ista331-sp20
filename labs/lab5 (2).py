import math, os, string, random
from sklearn.metrics import confusion_matrix 
from sklearn.feature_extraction import stop_words
from nltk.stem import SnowballStemmer



def get_data(paths):
    '''
    pass in the paths to the fiction and nonfiction directories as strings
    for each path, use os.listdir to access the .txt files in the directories
    read the contents of the .txt files into the X list, and add an associated 0 or 1 to the y list (depending on path you're getting the text from)
    the first list - X - should contain strings containing the text from each .txt file
    the second list - y - should contain a number representing genre(1 being fiction or positive, 0 being nonfiction or negative)
    return the lists
    '''
    #edits needed
    X = []
    y = []
    
    
    return X,y
    
def shuffle_data(X,y):
    '''
    use the random module to shuffle your X and y lists
    remember that the values in the X and y lists are associated, ex. X[0] matches y[0] and X[132] matches y[132] and so on...
    use a method that rearranges the lists so that they are shuffled in the same way
    '''
   

class NaiveBayesClassifier:
    def __init__(self, data, k = 0.5, max_words = 50):
        '''
        pass in data set to be analyzed
        crete the stemmer attribute using the snowball stemmer set to 'english'
        create the max words attribute using the max words argument
        use the data to create the X and y (features and labels lists)
        this class should have six attributes: X, y, a count of fiction books, a count of nonfiction books, a ratio of fiction/all and nonfiction/all
        books, and a dictionary of conditional probabilites for each feature (where A is genre (fiction/nonfiction) and B is the feature)
        '''
        #edits needed
        
        #fetch and divide data
        self.X = data[0]
        self.y = data[1]
        
        #get counts of fiction and not fiction tomatoes
        self.fiction_count = sum(self.y)
        self.nonfiction_count = len(self.y) - self.fiction_count
        self.fiction_ratio = self.fiction_count/len(self.y)
        self.nonfiction_ratio = self.nonfiction_count/len(self.y)
        
        #get the probabilties of features given fiction/not fiction
        self.featprobs = self.count_features()
        for feature in self.featprobs:
            self.featprobs[feature][0] = (self.featprobs[feature][0] + k)/(self.fiction_count + 2 * k)
            self.featprobs[feature][1] = (self.featprobs[feature][1] + k)/(self.nonfiction_count + 2 * k)
            
    def tokenize(self,text):
        '''
        takes the text from a book and tokenizes it
        make the tokens lower case
        removes all punctuation, 'n't',  and digits and stems the remaining words
        return the stemmed token vector as a set
        '''
        
            
    def count_features(self):
        '''
        create a dictionary the maps features to a two element list
        the list represetns two counts: one for number of fiction books with this feature, and one for nonfiction books with this feature
        '''
        #edits needed
        featdict = {}
     
        for i in range(len(self.X)):
            for feature in self.X:
                if feature not in featdict:
                    featdict[feature] = [0,0]
                if self.y[i] == 1:
                    featdict[feature][0] += 1
                else:
                    featdict[feature][1] += 1
        return featdict
        
    def get_tokens(self, text):
        '''
        Takes a feature vector (the tokenized text) and returns a random sample of tokens from the vector
        the number or tokens should be equal to max_word, or the length of the vector if it is less than max_word
        '''
  
        
    def fiction_prob(self, text):
        '''
        takes text from a book and tokenizes it, then gets a number of random tokens equal to max_word
        determines the probability of the instance being fiction with the given features
        create a sum of the logs of the probabilities of fiction and nonfiction given the feature
        use the exponents of the logs in bayes theorem and return the result
        '''
        #edits needed
        log_p_if_f = 0.0
        log_p_if_n = 0.0
        for feature in fv:
            if feature in self.featprobs:
                p_if_fic, p_if_non = self.featprobs[feature]
                log_p_if_f += math.log(p_if_fic)
                log_p_if_n += math.log(p_if_non)
        p_if_f = math.exp(log_p_if_f)
        p_if_n = math.exp(log_p_if_n)
        return (p_if_f * self.fiction_ratio) / (p_if_f * self.fiction_ratio + p_if_n * self.nonfiction_ratio)
        
    def classify(self, fv):
        '''
        takes a feture vector
        classifies the feature vector as fiction or nonfiction using fiction_prob
        fiction/nonfiction scale ranges from 0 (nonfiction) to 1 (fiction) with .5 being the median value on the scale
        returns True if fiction and False if nonfiction
        '''
        return self.fiction_prob(fv) >= 0.5
        
    def predict(self, X):
        '''
        note that this function takes an argument X
        although we will be using self to classify the feature vectors, the vectors we are classifying are ones that our classifier has not seen
        '''
        predictions = []
        for fv in X:
            predictions.append(self.classify(fv))
        return predictions

def main():
    '''
    determine your paths to the data
    get data + split data into train and test sets
    make and then run (train and test) naive bayes classifier
    get confusion matrix and transpose
    '''
    #edits needed
    
    #shuffle the data and split into train and test
    
    splitpoint = int(len(data[0])*.80)
    traindata = (data[0][:splitpoint],data[1][:splitpoint])
    testdata = (data[0][splitpoint:],data[1][splitpoint:])
    
    
    #create the classifier, passing in the train data
    #passing in the trian data means it will use the train data to create the counts and probs attributes used to make predictions
    classifier = NaiveBayesClassifier(traindata)
    
    #now, make predictions about the test data
    predictions = classifier.predict(testdata[0])
   
    #create the confusion matrix using predictions and the y labels from the test data
    cf = confusion_matrix(testdata[1], predictions)
    print(cf.T)
    
    


if __name__ == '__main__':
    main()
    
    