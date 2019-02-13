import os
import scipy.sparse as sp
import numpy as np
import pandas as pd
# Fetching and formatting datasets

dir = os.path.dirname(__file__)


def fetch_books():

    path = os.path.join(dir, "csvfiles", "BX-Books.csv")
    df = pd.read_csv(path, delimiter=";",
                     error_bad_lines=False,
                     encoding='ISO-8859-1',
                     usecols=["ISBN",
                              "Book-Title",
                              "Book-Author",
                              "Year-Of-Publication",
                              "Publisher"],
                     low_memory=False
                     )
    return df


def fetch_book_info(df, isbn):
    return df.loc[df["ISBN"] == isbn]


# Row: users, Column: books, value: ratings
def fetch_ratings(df):

    row, col = [], []

    df = df.reset_index(drop=True)

    # In order
    users = df.iloc[:, 0].astype('int32').to_numpy()
    books = df.iloc[:, 1].to_numpy()
    ratings = df.iloc[:, 2].astype('int32').to_numpy()

    # Initialize matrix row, col and value
    # In sequential order, but a[i] might be correspond to b[i]
    set_users = np.unique(users)
    set_books = np.unique(books)

    for user in users:
        row.append(set_users.searchsorted(user))
    for book in books:
        col.append(set_books.searchsorted(book))

    coo = sp.coo_matrix((ratings, (row, col)))

    print(repr(coo))

    dictionary = {'spr_mtrx': coo, 'book_id': books, 'users': users}

    return dictionary
