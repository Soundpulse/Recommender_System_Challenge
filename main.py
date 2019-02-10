import numpy as np
from lightfm import LightFM
from data import fetch_ratings
from scipy.sparse import hstack

# training and testing data

dfs_train = fetch_ratings("BX-Ratings-Train.csv")
dfs_test = fetch_ratings("BX-Ratings-Test.csv")

# create model (weighted approximate rank pairwase)
model = LightFM(loss='warp')

# train model
model.fit(dfs_train, epochs=30, num_threads=2)


def sample_recommendation(model, data, user_ids):

    n_users, n_items = data.shape

    for user_id in user_ids:

        # known_positives = dfs_train.indices
        known_positives = data.tocsr()[user_id].indices

        # books our model predicts he'll like
        scores = model.predict(user_id, np.arange(n_items))

        # rank them in order
        top_scores = np.argsort(-scores)

        # print out the resultsdata['train']
        print("User %s" % user_id)
        print("     Known positives:")

        for x in known_positives[:3]:
            print("         %s" % x)
        print("     Recommended:")

        for x in top_scores[:3]:
            print("         %s" % x)


sample_recommendation(model, dfs_train, [2])
