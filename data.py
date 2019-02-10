import os
import pandas as pd
from scipy.sparse import csr_matrix
from pandas.api.types import CategoricalDtype

# Fetching and formatting datasets

dir = os.path.dirname(__file__)


# TODO List
# min_rating
def fetch_ratings(file):
    path = os.path.join(dir, "csvfiles", file)
    df = pd.read_csv(path, sep=";", header=None)

    # Not sparse
    df.columns = ['uid', 'isbn', 'rating']
    df.pivot(index='uid', columns='isbn', values='rating')

    # sparse
    uid_c = CategoricalDtype(sorted(df.uid.unique()), ordered=True)
    isbn_c = CategoricalDtype(sorted(df.isbn.unique()), ordered=True)

    row = df.uid.astype(uid_c).cat.codes
    col = df.isbn.astype(isbn_c).cat.codes
    sparse_matrix = csr_matrix((df['rating'], (row, col)),
                               shape=(uid_c.categories.size,
                                      isbn_c.categories.size))
    print(repr(sparse_matrix))
    # dfs = pd.SparseDataFrame(sparse_matrix,
    #                         index=uid_c.categories,
    #                         columns=isbn_c.categories,
    #                         default_fill_value=0)
    return sparse_matrix


fetch_ratings("BX-Ratings-Test.csv")
