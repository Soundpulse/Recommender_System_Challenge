import matplotlib.pyplot as plt
import numpy as np
from lightfm import LightFM
from lightfm.evaluation import precision_at_k
from lightfm.evaluation import auc_score
from data import fetch_ratings
from data import fetch_books
from data import fetch_book_info
from split import build_dataset
import scipy as sp


def sample_recommendation(model, data, user_ids):

    # Initialize variables
    n_users, n_books = data['spr_mtrx'].shape

    coo = data['spr_mtrx']
    users = data['users']
    bid = data['book_id']
    csr = coo.tocsr()

    # load book data
    books = fetch_books()

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
            info = fetch_book_info(books, bid[known_positives[-i]])
            s = known_score[-i]
            if info.empty:
                print("        Score: %s, Book: not in database :( ISBN: %s " %
                      (s, bid[known_positives[-i]]))
            else:
                print("        Score: %s, Book: %s ISBN: %s" %
                      (s, info.iloc[0, 1], info.iloc[0, 0]))

        # books our model predicts the user will like
        scores = model.predict(user_id, np.arange(n_books))

        # rank them in order (only take the top 5 <O(N) Time>)
        predict_recommendations = np.argpartition(scores, -6)[-6:]
        predict_recommendations = predict_recommendations[np.argsort(
            scores[predict_recommendations])]

        predict_score = scores[predict_recommendations]

        # print Recommendations
        print("     Recommended:")
        for i in range(1, 6):
            info = fetch_book_info(books,
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


def train(models, train, test):
    highest_test_precision = 0
    best_model = None

    for model in models:
        # train model
        print("Fitting...")
        model.fit(train['spr_mtrx'], epochs=30)

        # zero indexing :)
        # /***********************************************\
        # | PLACE USER_ID HERE FOR PRINTING AND TESTING!! |
        # \***********************************************/
        # sample_recommendation(model, train, [])

        # testing accuracy
        train_precision = precision_at_k(model,
                                         train['spr_mtrx'],
                                         k=10).mean()
        test_precision = precision_at_k(model,
                                        test['spr_mtrx'],
                                        k=10).mean()

        train_auc = auc_score(model,
                              train['spr_mtrx']).mean()
        test_auc = auc_score(model,
                             test['spr_mtrx']).mean()

        if model == model_w:
            print("WARP:")
        if model == model_l:
            print("Logistic:")
        if model == model_b:
            print("BPR:")

        # printout statements
        print('Precision - Train: %.3f, Test: %.3f.' % (train_precision,
                                                        test_precision))
        print('AUC - Train: %.3f, Test: %.3f.' % (train_auc, test_auc))

        if test_precision > highest_test_precision:
            highest_test_precision = test_precision
            best_model = model

    if best_model == model_w:
        print("Best Model: WARP, test precision: %.8f" %
              highest_test_precision)
    if best_model == model_l:
        print("Best Model: Logistic, test precision: %.8f" %
              highest_test_precision)
    if best_model == model_b:
        print("Best Model: BPR, test precision: %.8f" %
              highest_test_precision)

    sample_recommendation(best_model, train, range(1, 10, 1))


# main program
# training and testing data
print("Fetching Data...")

print("Building dataset...")
dataset = build_dataset("BX-Book-Ratings.csv", min_rating=1)
dfs_train = fetch_ratings(dataset['train'])
dfs_test = fetch_ratings(dataset['test'])

# Generating Heat Map
print("Generating Heat Map...")

dm = dfs_train['spr_mtrx'].todense()

plt.imshow(dm[1:10000, 1:10000], cmap='hot')
plt.show()

print("Printing representations:")

print(repr(dfs_train))
print(repr(dfs_test))

# create model
print("Training Model...")
model_w = LightFM(loss='warp')
model_l = LightFM(loss='logistic')
model_b = LightFM(loss='bpr')
models = [model_w, model_l, model_b]

train(models, dfs_train, dfs_test)
