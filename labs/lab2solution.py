import pandas as pd, numpy as np, sqlite3


def genres_ratings_titles(conn):
    '''
    use the given connection object along with sql queries to select title, rating, and genre columns from the table
    return a pandas dataframe with movie titles as the index and two columns, one with the genre of the titles and one with the rating
    don't worry about nan's for this assignment
    '''
    c = conn.cursor()
    query = 'SELECT title, content_rating, genre FROM movies;'
    c.execute(query)
    indx = []
    values = []
    for row in c.fetchall():
        indx.append(row[0])
        values.append(row[1:])
    return pd.DataFrame(data = values, index = indx, columns = ['rating', 'genre'])
    


def genre_rating_matrix(df):
    '''
    use the dataframe you created to create a sparse matirx where the row labels are genres and the columns are ratings
    initialize all of the values as 0
    traverse through your dataframe and incrament the appropirate value by 1 based off of the rating and genre of the title
    return your sparse matrix
    '''
    mat_vals = np.zeros(shape = (len(set(df['genre'])),len(set(df['rating']))))
    mat = pd.DataFrame(data = mat_vals, index = set(df['genre']), columns = set(df['rating']))
    
    for title in df.index:
        mat.loc[df.loc[title,'genre'],df.loc[title,'rating']] += 1
    
    return mat


def get_cond_prob(mat):
    '''
    ask the use which genre and rating they would like to calculate a conditional probability for
    use your matrix and the conditional probability function to find the conditional probability of a movie's rating given it's genre
    what is your A? what is your B?
    return the conditional probability
    '''
    
    #A is rating = given rating genre and B is genre = given
    genre = input('What genre would you like to inspect: ').capitalize()
    rating = input('What rating would you like to inspect: ').upper()
    if rating == 'NAN':
        rating = 'nan'
    
    A_and_B = mat.loc[genre,rating]
    B = sum(mat.loc[genre,:])
    
    return A_and_B/B

def main():
    '''
    connect to movies.db and test your functions
    '''
    conn = sqlite3.connect('movies.db')
    conn.row_factory = sqlite3.Row
    
    df = genres_ratings_titles(conn)
    print(df)
    mat = genre_rating_matrix(df)
    
    cond_prob = get_cond_prob(mat)
    
    #print(cond_prob)



if __name__ == '__main__':
    main()
    
    