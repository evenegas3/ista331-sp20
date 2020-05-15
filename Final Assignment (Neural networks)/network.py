import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation

def get_data():
    """
    Function reads 'seeds.csv' into a dataframe. Then, use np.random.choice to select 105 rows to be the training set.
    Once you have split the data frame into train and test frames, Keras requires that you do a little bit more work to get the data frames prepped.

    PARAMETERS: None

    RETURNS:training_x -- dataset of x to train from 'seeds.csv'
            training_y -- dataset of y to train from 'seeds.csv'
            testing_x -- dataset of x to test from 'seeds.csv'
            testing_y -- dataset of y to test from 'seeds.csv'
    """
    initial_data = pd.read_csv('seeds.csv')
    choices = np.random.choice(initial_data.index, size=105, replace=False)
    train, test = initial_data.iloc[choices], initial_data.drop(choices, axis=0)
    
    training_x, testing_x = train.drop('class', axis=1), test.drop(columns=['class'])
    training_x, testing_x = training_x.values, testing_x.values

    training_y, testing_y = train['class'] - 1, test['class'] - 1
    training_y, testing_y = keras.utils.to_categorical(training_y), keras.utils.to_categorical(testing_y)

    return training_x, training_y, testing_x, testing_y

def setup_network(layers, active_function):
    """
    This function takes a list of integers and a string representing an activation function (one of ’sigmoid’, ’tanh’, or ’relu’;
    initializes a neural network model. The integers represent the sizes of Dense network layers.

    PARAMETERS: layers -- a list of integers, used to represent the sizes of the Dense layers
                active_function -- a string, the name of the activation, to initialize the model.

    RETURNS: model -- a model of either ’sigmoid’, ’tanh’, or ’relu’, with added dense layers from argument list.
    """
    model = Sequential()
    for i in range(0, len(layers)):
        if i == 0:
            model.add(Dense(layers[i], input_shape=(7,), activation=active_function))
        else:
            model.add(Dense(layers[i], activation=active_function))

    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def train_network(model, train_x, train_y, epoch_count):
    """
    This function takes a network that has been created by setup network, a training X, a training y,
    and a number of epochs. Train the network using the X and y data, a batch size of 16, and the given number of epochs.
    You can return None.

    PARAMETERS: model -- The model returned from setup_network()
                train_x -- a dataset of training x data
                train_y -- a dataset of training y data
                epoch_count -- an integer, number of epochs used for testing accuracy

    RETURNS: None
    """
    model.fit(train_x, train_y, batch_size=16, epochs=epoch_count)
    return None

def test_network(train_network, test_x, test_y):
    """
    This function takes a trained network, a testing X, and a testing y.
    Call the evaluate method from the network and pass it the testing X and testing y, and a batch size of 1.
    Return the training accuracy.

    PARAMETER: train_network -- a trained data fitted model (batchsize=16, epochs=25)
               test_x -- a dataset of x data
               test_y -- a dataset of y data

    RETURNS: acc -- a float, the accuracy from trained_network evaluated
    """
    score, acc = train_network.evaluate(test_x, test_y, batch_size=1)
    return acc
    

def main():
    int_list = [512, 256, 128, 46]
    function_name = 'relu'
    epochs = 25

    train_x, train_y, test_x, test_y = get_data()
    model = setup_network(int_list, function_name)

    train_network(model, train_x, train_y, epochs)
    train_network(model, train_x, train_y, epochs)
    train_network(model, train_x, train_y, epochs)
    train_network(model, train_x, train_y, epochs)
    train_network(model, train_x, train_y, epochs)

    acc = test_network(model, test_x, test_y)

    print('Network architecture: {}'.format(int_list))
    print('Activation function: {}'.format(function_name))
    print('Number of epochs: {}'.format(epochs))
    print("Test accuracy: {}".format(acc))

main()


