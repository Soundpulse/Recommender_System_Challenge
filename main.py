import numpy as np
from lightfm import LightFM
from data import fetch_ratings

# training and testing data

dfs_train = fetch_ratings("BX-Ratings-Train.csv", 5)
dfs_test = fetch_ratings("BX-Ratings-Test.csv", 5)

# create model (weighted approximate rank pairwase)
model = LightFM(loss='warp')

# train model
model.fit(dfs_train['spr_mtrx'], epochs=30, num_threads=2)


def sample_recommendation(model, data, user_ids):

    n_users, n_books = data['spr_mtrx'].shape
    coo = data['spr_mtrx']
    csr = coo.tocsr()

    for user_id in user_ids:
        # known_positives
        # TODO
        # Better sort for csr
        row = csr.getrow(user_id).toarray()[0].ravel()
        known_score = row.argsort()
        known_positives = row[row.argsort()]

        for i in range(1, 6):
            print("        Score: %s, Book: %s" %
                  (known_positives[-i], known_score[-i]))

        # books our model predicts he'll like
        scores = model.predict(user_id, np.arange(n_books))

        # rank them in order (only take the top 5 <O(N) Time>)
        predict_recommendations = np.argpartition(scores, -6)[-6:]
        predict_recommendations = predict_recommendations[np.argsort(
            scores[predict_recommendations])]
        predict_score = scores[predict_recommendations]

        print("     Recommended:")
        for i in range(1, 6):
            print("        Score: %s, Book: %s" %
                  (predict_score[6 - i], predict_recommendations[6 - i]))

        print(".=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+.")


sample_recommendation(model, dfs_train, range(1, 50, 1))
