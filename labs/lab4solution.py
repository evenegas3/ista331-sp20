import math
from sklearn.metrics import confusion_matrix



def get_data(fname):
    '''
    pass in .csv file with four columns - shape, size, color, deformed status
    read contents into a two lists
    the first list - X - should contain tuples containing the features of the tomatoes (shape, size, and color)
    the second list - y - should contain a number representing deformed status (1 being deformed or positive, 0 being deformed or negative)
    return the lists
    '''
    X = []
    y = []
    file = open(fname)
    file.readline()
    for line in file:
        line = line.split(',')
        X.append(tuple(line[0:3]))
        y.append(int(line[3]))
    file.close()
    return X,y

class NaiveBayesClassifier:
    def __init__(self, data, k = 0.5):
        '''
        pass in data set to be analyzed
        use the data to create the X and y (features and labels lists)
        this class should have six attributes: X, y, a count of deformed tomatoes, a count of normal tomatoes, a ratio of normal/deformed tomatoes,
        and a dictionary of conditional probabilites for each feature (where A is deformed status and B is the feature)
        '''
        #fetch and divide data
        self.X = data[0]
        self.y = data[1]
        
        #get counts of deformed and not deformed tomatoes
        self.deformed_count = sum(self.y)
        self.normal_count = len(self.y) - self.deformed_count
        self.deformed_ratio = self.deformed_count/len(self.y)
        self.normal_ratio = self.normal_count/len(self.y)
        
        #get the probabilties of features given deformed/not deformed
        self.featprobs = self.count_features()
        for feature in self.featprobs:
            self.featprobs[feature][0] = (self.featprobs[feature][0] + k)/(self.deformed_count + 2 * k)
            self.featprobs[feature][1] = (self.featprobs[feature][1] + k)/(self.normal_count + 2 * k)
            
    def count_features(self):
        '''
        create a dictionary the maps features to a two element list
        the list represetns two counts: one for number of deformed tomatoes with this feature, and one for normal tomatoes with this feature
        '''
        featdict = {}
        for i in range(len(self.X)):
            for feature in self.X[i]:
                if feature not in featdict:
                    featdict[feature] = [0,0]
                if self.y[i] == 1:
                    featdict[feature][0] += 1
                else:
                    featdict[feature][1] += 1
        return featdict
        
    def deformed_prob(self, fv):
        '''
        takes a feature vector
        determines the probability of the instnace being deformed with the given features
        create a sum of the logs of the probabilitys of deforemd and normal given the feature
        use the exponents of the logs in bayes theorem and return the result
        '''
        log_p_if_d = 0.0
        log_p_if_n = 0.0
        for feature in fv:
            if feature in self.featprobs:
                p_if_def, p_if_norm = self.featprobs[feature]
                log_p_if_d += math.log(p_if_def)
                log_p_if_n += math.log(p_if_norm)
        p_if_d = math.exp(log_p_if_d)
        p_if_n = math.exp(log_p_if_n)
        return (p_if_d * self.deformed_ratio) / (p_if_d * self.deformed_ratio + p_if_n * self.normal_ratio)
        
    def classify(self, fv):
        '''
        takes a feture vector
        classifies the feature vector as deformed or normal using deformed_prob
        deformed/normal scale ranges from 0 (normal) to 1 (deformed) with .5 being the median value on the scale
        returns True if deforemed and False if not deformed
        '''
        return self.deformed_prob(fv) >= 0.5
        
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
    get data + split data into train and test sets
    make and then run (train and test) naive bayes classifier
    get confusion matrix and transpose
    '''
    fname = 'tomatoes.csv'
    data = get_data(fname)
    
    #split into train and test
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
    
    