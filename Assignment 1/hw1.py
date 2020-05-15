"""
Erick Venegas
ISTA 331
02-05-20
hw1.py
Sean Current

This homework will introduce you to the paradigm-altering concept of recommender systems.
Recommender systems enable targeted advertising, which has changed the face of business
across the planet and made Amazon into a global hegemon.
"""

import pandas as pd, numpy as np, random, sqlite3

def get_recommendation(id, spm, isbn_history, conn):
    """
    This function takes a customer id, a sparse probability matrix, purchase history ISBN list, and a connection.
    Use the previous function to randomly grab a book from the customer's most recent purchase. Get the customer's name.
    Get the two books most similar to the recently purchased book, not including any books already purchased by the customer.

    PARAMETERS: id -- a string, a customer's id
                spm -- a sparse probability matrix
                isbn_history -- a list, purchase isbn history(strings)
                conn -- a connection object to the SQL database
    
    RETURNS: output_string -- a string, displaying user's name and recommended books for user
    """
    c = conn.cursor()
    book = get_recent(id, conn)
    info = c.execute("SELECT first, last FROM Customers WHERE cust_id={};".format(str(id)))
    output_string = ""

    for row in info.fetchall():
        first, last = row[0], row[1]
        output_string = "Recommendations for {} {}".format(first, last) + "\n"

    output_string += "-" * (len(output_string) - 1) + "\n"

    for isbn in spm[book]:
        if isbn not in isbn_history:
            for row in c.execute("SELECT book_title FROM Books WHERE isbn={};".format(isbn)).fetchall():
                output_string += row[0] + "\n"

    return output_string

def get_recent(id, conn):
    """
    This function takes a customer id and a connection. It returns an ISBN chosen randomly from the customers most recent order.
    Make a list of the ISBN's in the most recent order and use random.randrange to randomly index into the list to grab a book.

    PARAMETERS: id -- a string, a customers id
                conn -- a connection object to the SQL database

    RETURNS: a string value of a book's isbn
    """
    c = conn.cursor()
    isbn_list, isbn = [], None
    
    for row in c.execute("SELECT order_num FROM Orders WHERE cust_id={};".format(str(id))).fetchall():
        isbn = row[0]

    for row in c.execute("SELECT isbn FROM OrderItems WHERE order_num ={};".format(str(isbn))).fetchall():
        isbn_list.append(row[0])

    return isbn_list[random.randrange(len(isbn_list))]

def purchase_history(id, isbn_list, conn):
    """
    This function takes a customer id, a list of ISBN's that the customer has purchased,
    and a connection to the db, and returns a string containing the customer's purchase history as titles

    PARAMETERS: id -- an integer, the customer id
                isbn_list -- a list of isbn string values
                conn -- a connection object to the SQL database
    
    RETURNS: message -- a string consisting of customer's purchase history as titles
    """
    c = conn.cursor()
    message = "Purchase history for "

    for row in c.execute("SELECT * FROM Customers;").fetchall():
        if row[0] == id:
            message += "{} {}".format(row[2], row[1])

    message += "\n" + "-" * len(message) + "\n"

    for b in isbn_list:
        for row in c.execute("SELECT isbn, book_title FROM Books;").fetchall():
            if row[0] == b:
                message += row[1] + "\n"

    return message + "-" * 40 + "\n"

def get_cust_id(conn):
    """
    This function takes a connection object and returns an integer customer id or None,
    depending upon user input. Grab the customer info from the db and print it

    PARAMETERS: conn -- a connection object to the SQL database.

    RETURNS: None or integer
    """
    c = conn.cursor()
    info = c.execute("SELECT * FROM Customers;")
    print("CID       Name")
    print("-----     -----")

    for row in info.fetchall():
        print("    {}     {}, {}".format(row[0], row[1], row[2]))

    print("---------------")
    user_input = input("Enter customer number or enter to quit: ")
    if user_input == "":
        return None

    return int(user_input)



def sparse_p_matrix(p_matrix):
    """
    Maps ISBN's to lists of ISBN's of other books sorted in descending order
    of likelihood that the other book was purchased if the key was purchased.
    Now that I thought of this way, we could map to index objects or arrays.
    """
    spm = {}
    for book in p_matrix.index:
        spm[book] = list(p_matrix.loc[book].sort_values(ascending=False, kind='mergesort')[:15].index)
    return spm

def make_probability_matrix(cm):
    """
    This function takes a count matrix and returns a conditional probability matrix

    PARAMETERS: cm -- conditional matrix

    RETURNS: cpm -- conditional probability matrix containing isbn
    """
    cpm = cm.copy()

    for row in cm.index:
        for col in cm.columns:
            if row == col:
                cpm.loc[row, col] = -1
            else:
                cpm.loc[row, col] = cm.loc[row, col] / cm.loc[row, row]

    return cpm

def fill_count_matrix(empty_matrix, purchase_matrix):
    """
    This function takes an empty count matrix and a purchase matrix and fills the count matrix.
    Go through each customer's list of ISBN's. For each ISBN in the list, increment the appropriate spot on the diagonal.
    We are keeping track of the number of users who have purchased each book on the diagonal.

    PARAMETERS: empty_matrix -- a zero filled matrix
                purchase_matrix -- number of times a book as been purchased
    
    RETURNS: N/A
    """
    for k, v in purchase_matrix.items():
        for val in v:
            empty_matrix.loc[val, val] += 1
    
    for k, v in purchase_matrix.items():
        for i in range(0, len(v) - 1):
            first_val = v.pop(0)
            rest = v
            empty_matrix.loc[first_val, rest] += 1
            empty_matrix.loc[rest, first_val] += 1

def get_empty_count_matrix(conn):
    """
    This function takes a connection object to a bookstore db and returns a DataFrame
    with index and columns that are the ISBN's of the books available.

    PARAMETERS: conn -- a connection object for the SQL database.

    RETURNS: df -- a dataframe with isbn as index and columns names, zeroes throughout matrix
    """
    c = conn.cursor()
    l = []

    for row in c.execute("SELECT isbn from books;").fetchall():
        l.append(row[0])
    
    df = pd.DataFrame(0, index=l, columns=l)

    return df

def get_purchase_matrix(conn):
    """
    This function takes a connection object to a bookstore transaction db and returns a dictionary that maps customer
    id's to a sorted lists of books they have purchased (i.e. ISBN's) without duplicates.
    It must be sorted for the test to work. For the small db, it looks like this

    PARAMETERS: conn -- a connection object for the SQL database

    RETURNS: d -- a dictionary, maps customer id's to a sorted lists of books they have purchased
    """

    c = conn.cursor()
    query = "SELECT * from CUSTOMERS join orders on customers.cust_id = orders.cust_id join orderitems on orders.order_num = orderitems.order_num;"
    c.execute(query)

    d = {}
    for row in c.fetchall():
        user_id, isbn = row[0], row[7]

        if user_id not in d:
            d[user_id] = [isbn]
        else:
            d[user_id].append(isbn)
    
    for k, v in d.items():
        v = v.sort()


    return d

def isbn_to_title(conn):
    c = conn.cursor()
    query = 'SELECT isbn, book_title FROM Books;'
    return {row['isbn']: row['book_title'] for row in c.execute(query).fetchall()}

def select_book(itt):
    isbns = sorted(itt)
    print('All books:')
    print('----------')
    for i, isbn in enumerate(isbns):
        print(' ', i, '-->', isbn, itt[isbn][:60])
    print('-' * 40)
    selection = input('Enter book number or return to quit: ')
    return isbns[int(selection)] if selection else None
    
def similar_books(key, cm, pm, itt, spm): # an isbn, count_matrix, p_matrix, isbn_to_title
    bk_lst = []
    for isbn in cm.columns:
        if key != isbn:
            bk_lst.append((cm.loc[key, isbn], isbn))
    bk_lst.sort(reverse=True)
    print('Books similar to', itt[key] + ':')
    print('-----------------' + '-' * (len(itt[key]) + 1))
    for i in range(5):
        print(str(i) + ':')
        print(' ', bk_lst[i][0], '--', itt[bk_lst[i][1]][:80])
        print('  spm:', itt[spm[key][i]][:80])
        print('  p_matrix:', pm.loc[key, bk_lst[i][1]])
        
    
def main1():
    conn = sqlite3.connect('small.db')
    conn.row_factory = sqlite3.Row
    purchase_matrix = get_purchase_matrix(conn)
    count_matrix = get_empty_count_matrix(conn)
    fill_count_matrix(count_matrix, purchase_matrix)
    p_matrix = make_probability_matrix(count_matrix)
    spm = sparse_p_matrix(p_matrix)
    ###
    # itt = isbn_to_title(conn)
    # selection = select_book(itt)
    # while selection:
    #     similar_books(selection, count_matrix, p_matrix, itt, spm)
    #     input('Enter to continue:')
    #     selection = select_book(itt)
    #####
    cid = get_cust_id(conn)
    while cid:
        print()
        titles = purchase_history(cid, purchase_matrix[cid], conn)
        print(titles)
        print(get_recommendation(cid, spm, purchase_matrix[cid], conn))
        input('Enter to continue:')
        cid = get_cust_id(conn)
    
def main2():
    conn = sqlite3.connect('bookstore.db')
    conn.row_factory = sqlite3.Row
    
    purchase_matrix = get_purchase_matrix(conn)
    print('*' * 20, 'Purchase Matrix', '*' * 20)
    print(purchase_matrix)
    print()
    
    count_matrix = get_empty_count_matrix(conn)
    print('*' * 20, 'Empty Count Matrix', '*' * 20)
    print(count_matrix)
    print()
    
    fill_count_matrix(count_matrix, purchase_matrix)
    print('*' * 20, 'Full Count Matrix', '*' * 20)
    print(count_matrix)
    print()
    
    p_matrix = make_probability_matrix(count_matrix)
    print('*' * 20, 'Probability Matrix', '*' * 20)
    print(p_matrix)
    print()
    
    spm = sparse_p_matrix(p_matrix)
    print('*' * 20, 'Sparse Probability Matrix', '*' * 20)
    print(spm)
    print()
    
    ######
    itt = isbn_to_title(conn)
    print('*' * 20, 'itt dict', '*' * 20)
    print(itt)
    print()
    
    """
    selection = select_book(itt)
    while selection:
        similar_books(selection, count_matrix, p_matrix, itt, spm)
        input('Enter to continue:')
        selection = select_book(itt)
    ######
    cid = get_cust_id(conn)
    while cid:
        print()
        titles = purchase_history(cid, purchase_matrix[cid], conn)
        print(titles)
        print(get_recommendation(cid, spm, purchase_matrix[cid], conn))
        input('Enter to continue:')
        cid = get_cust_id(conn)
    """
if __name__ == "__main__":
    main1()
    
    
    
    
    
    
    
    
