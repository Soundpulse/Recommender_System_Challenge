import os
import scipy.sparse as sp
import csv
# Fetching and formatting datasets

dir = os.path.dirname(__file__)

# Data to create our coo_matrix
data, i, j = [], [], []

users, books = [], []


# Row: users, Column: books, value: ratings
def fetch_ratings(file, min_rating=0):

    # Ratings can only be integers
    min_rating = int(min_rating)

    path = os.path.join(dir, "csvfiles", file)
    f = open(path)
    reader = csv.reader(f)

    for line in reader:
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
        'books': books,
        'users': users
    }

    return dictionary
