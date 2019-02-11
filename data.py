import os
import scipy.sparse as sp
import csv
import pandas as pd
# Fetching and formatting datasets

dir = os.path.dirname(__file__)

# Data to create our coo_matrix
data, i, j = [], [], []

users, books = [], []


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
def fetch_ratings(file, min_rating=0):

    path = os.path.join(dir, "csvfiles", file)
    f = open(path, errors='ignore', encoding='ISO-8859-1')
    reader = csv.reader(f)
    counter = 0

    for line in reader:
        counter = counter + 1
        if(counter % 10000) == 0:
            print("Fetched: %d" % counter)
        temp = [x for x in line[0].replace('"', '').split(';')]
        user = int(temp[0])
        isbn = temp[1]
        rating = int(temp[2])

        # Assign user in users
        if user not in users:
            users.append(user)

        # TODO:
        # IMPLEMENT NAME AND INFORMATION OF BOOK GIVEN ISBN
        if isbn not in books:
            books.append(isbn)

        if rating >= min_rating:
            data.append(rating)
            i.append(users.index(user))
            j.append(books.index(isbn))

    coo = sp.coo_matrix((data, (i, j)))

    dictionary = {
        'spr_mtrx': coo,
        'book_id': books,
        'users': users
    }

    return dictionary
