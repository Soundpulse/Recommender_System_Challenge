import numpy as np
from lightfm import LightFM
from lightfm.evaluation import precision_at_k
from lightfm.evaluation import auc_score
import data as d

# TODO List
# Better sort for csr


def sample_recommendation(model, data, user_ids):

    # Initialize variables
    print("Initializing...")
    n_users, n_books = data['spr_mtrx'].shape

    coo = data['spr_mtrx']
    users = data['users']
    bid = data['book_id']
    csr = coo.tocsr()

    # load book data
    books = d.fetch_books()

    print("Complete.")
    print(".=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+.")

    for user_id in user_ids:

        # sort the known positives
        row = csr.getrow(user_id).toarray()[0].ravel()
        known_positives = row.argsort()
        known_score = row[row.argsort()]

        # printout statements
        # user's name has been anonymized so the numbers are used.
        print("For User %d: " % users[user_id])

        # not enough user input (User has not liked enough books)
        # prevent giving users books that he don't like
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
                print('        Score: %s, Book: not in database :('
                      '  ISBN: %s' %
                      (s, bid[predict_recommendations[6 - i]]))
            else:
                print("        Score: %s, Book: %s ISBN: %s" %
                      (s, info.iloc[0, 1], info.iloc[0, 0]))

        print(".=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+.")


# main program
# training and testing data
print("Fetching Data...")
# dfs = d.fetch_ratings("BX-Book-Ratings.csv", 5)
dfs_train = d.fetch_ratings("BX-Ratings-Train.csv", 5)
dfs_test = d.fetch_ratings("BX-Ratings-Test.csv", 5)

# create model (weighted approximate rank pairwase)
print("Training Model...")
model_w = LightFM(loss='warp')
model_l = LightFM(loss='logistic')
model_b = LightFM(loss='bpr')
# train model
print("Fitting...")
model_w.fit(dfs_train['spr_mtrx'], epochs=50)
model_l.fit(dfs_train['spr_mtrx'], epochs=50)
model_b.fit(dfs_train['spr_mtrx'], epochs=50)

# zero indexing cause why not? :)
sample_recommendation(model_w, dfs_train, [])
sample_recommendation(model_l, dfs_train, [])
sample_recommendation(model_b, dfs_train, [])

# testing accuracy
models = [model_w, model_l, model_b]


for model in models:
    train_precision = precision_at_k(model, dfs_train['spr_mtrx'], k=10).mean()
    test_precision = precision_at_k(model, dfs_test['spr_mtrx'], k=10).mean()

    train_auc = auc_score(model, dfs_train['spr_mtrx']).mean()
    test_auc = auc_score(model, dfs_test['spr_mtrx']).mean()

    if model == model_w:
        print("WARP:")
    if model == model_l:
        print("Logistic:")
    if model == model_b:
        print("BPR:")

    print('Precision - Train: %.2f, Test: %.2f.' % (train_precision,
                                                    test_precision))
    print('AUC - Train: %.2f, Test: %.2f.' % (train_auc,
                                              test_auc))
