import pandas as pd, numpy as np, sqlite3


def genres_ratings_titles(conn):
    '''
    use the given connection object along with sql queries to select title, rating, and genre columns from the table
    return a pandas dataframe with movie titles as the index and two columns, one with the genre of the titles and one with the rating
    don't worry about nan's for this assignment
    '''
   


def genre_rating_matrix(df):
    '''
    use the dataframe you created to create a sparse matirx where the row labels are genres and the columns are ratings
    initialize all of the values as 0
    traverse through your dataframe and incrament the appropirate value by 1 based off of the rating and genre of the title
    return your sparse matrix
    '''
  

def get_cond_prob(mat):
    '''
    ask the use which genre and rating they would like to calculate a conditional probability for
    use your matrix and the conditional probability function to find the conditional probability of a movie's rating given it's genre
    what is your A? what is your B?
    return the conditional probability
    '''
    
   
def main():
    '''
    connect to movies.db and test your functions
    '''
    



if __name__ == '__main__':
    main()
    
    