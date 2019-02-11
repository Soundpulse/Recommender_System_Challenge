import numpy as np
from lightfm import LightFM
import data as d

# TODO List
# Better sort for csr

# training and testing data
# testing set available but I have no idea how to test this :/
dfs_train = d.fetch_ratings("BX-Ratings-Train.csv", 0)
dfs_test = d.fetch_ratings("BX-Ratings-Test.csv", 0)

# create model (weighted approximate rank pairwase)
model = LightFM(loss='warp')

# train model
model.fit(dfs_train['spr_mtrx'], epochs=30, num_threads=2)


def sample_recommendation(model, data, user_ids):

    # Initialize variables
    n_users, n_books = data['spr_mtrx'].shape

    coo = data['spr_mtrx']
    users = data['users']
    bid = data['book_id']
    csr = coo.tocsr()

    # load book data
    books = d.fetch_books()

    for user_id in user_ids:

        # sort the known positives
        row = csr.getrow(user_id).toarray()[0].ravel()
        known_positives = row.argsort()
        known_score = row[row.argsort()]

        # printout statements
        # user's name has been anonymized so the numbers are used.
        print("For User %d: " % users[user_id])

        # not enough user input (User has not liked enough books)
        if known_score[-1] <= 3:
            print("Insufficient Data. Generating Random Premutations...")

        # All info is fetched for easier access later
        # print known positives
        for i in range(1, 6):
            info = d.fetch_book_info(books, bid[known_positives[-i]])
            s = known_score[-i]
            if info.empty:
                print("        Score: %s, Book: not in database :( ISBN: %s " %
                      (s, bid[known_positives[-i]]))
            else:
                print("        Score: %s, Book: %s ISBN: %s" %
                      (s, info.iloc[0, 1], info.iloc[0, 0]))

        # books our model predicts he'll like
        scores = model.predict(user_id, np.arange(n_books))

        # rank them in order (only take the top 5 <O(N) Time>)
        predict_recommendations = np.argpartition(scores, -6)[-6:]
        predict_recommendations = predict_recommendations[np.argsort(
            scores[predict_recommendations])]

        predict_score = scores[predict_recommendations]

        # print Recommendations
        print("     Recommended:")
        for i in range(1, 6):
            info = d.fetch_book_info(books,
                                     bid[predict_recommendations[6 - i]])
            s = round(predict_score[6 - i], 2)
            if info.empty:
                print("        Score: %s, Book: not in database :(  ISBN: %s" %
                      (s, bid[predict_recommendations[6 - i]]))
            else:
                print("        Score: %s, Book: %s ISBN: %s" %
                      (s, info.iloc[0, 1], info.iloc[0, 0]))

        print(".=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+.")


# zero indexing cause why not? :)
sample_recommendation(model, dfs_train, range(0, 20, 1))
