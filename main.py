import numpy as np
from lightfm import LightFM
from data import fetch_ratings

# training and testing data

dfs_train = fetch_ratings("BX-Ratings-Train.csv", 5)
dfs_test = fetch_ratings("BX-Ratings-Test.csv", 5)

# create model (weighted approximate rank pairwase)
model = LightFM(loss='warp')

# train model
# model.fit(dfs_train, epochs=30, num_threads=2)


def sample_recommendation(model, data, user_ids):

    n_users, n_items = data['spr_mtrx'].shape
    coo = data['spr_mtrx']
    csr = coo.tocsr()

    for user_id in user_ids:

        # known_positives
        row = csr.getrow(user_id).toarray()[0].ravel()
        known_score = row.argsort()
        known_positives = row[row.argsort()]

        for i in range(1, 5):
            print("        Book: %s, Index: %s" %
                  (known_positives[-i], known_score[-i]))

        # books our model predicts he'll like
        # scores = model.predict(user_id, np.arange(n_items))

        # rank them in order
        # top_scores = csr[user_id, np.argsort(-scores)]

        # print out the resultsdata['train']


sample_recommendation(model, dfs_train, [2])
