import pickle

import faiss
import numpy as np
import pandas as pd
from scipy.sparse import load_npz

X = load_npz('saved_models/sparse_data.npz')

X = X.toarray()
y = X[:, -1]
X = X[:, :-1]

index = faiss.IndexFlatL2(X.shape[1])
index.add(X)

print(index.ntotal)

filename = 'saved_models/svd_model.sav'
algo = pickle.load(open(filename, 'rb'))


def get_similar_recipes_collab(recipe_id):
    recipe_index = np.where(y == recipe_id)[0][0]
    recipe_vector = X[recipe_index]
    # search for the nearest neighbors
    k = 1000
    D, I = index.search(np.array([recipe_vector]), k)
    recipe_ids = y[I[0]]
    unique_recipe_ids = np.unique(recipe_ids)
    unique_recipe_ids = unique_recipe_ids[unique_recipe_ids != recipe_id]
    return unique_recipe_ids.tolist()


def get_similar_recipes_svd(df, recipe_id):
    recipe_ratings = df[df.recipe_id == recipe_id]
    recipe_ratings['rating'] = recipe_ratings['rating'].astype(float)
    recipe_mean_rating = recipe_ratings['rating'].mean()

    test_set = [[user_id, recipe_id, recipe_mean_rating] for user_id in df.user_id.unique()]
    predictions = algo.test(test_set)

    # filter predictions for the target recipe
    target_predictions = [pred for pred in predictions if pred.iid == recipe_id]

    # sort target predictions by estimated rating
    target_predictions.sort(key=lambda x: x.est, reverse=True)

    top_n_recipes = [pred.uid for pred in target_predictions]
    return top_n_recipes


def get_similar_recipes_hybrid(df, recipe_id, n=10):
    collab = get_similar_recipes_collab(recipe_id)
    svd = get_similar_recipes_svd(df, recipe_id)
    print("Collab: ", len(collab))
    print("SVD: ", len(svd))
    # find the intersection of the two lists
    # similar_recipes = list(set(collab) & set(svd))
    similar_recipes = list(set(collab))
    similar_recipes = [int(x) for x in similar_recipes]
    print("Intersection: ", len(similar_recipes))
    similar_recipes = similar_recipes[:n]
    top_n_recipe_data = df[df.recipe_id.isin(similar_recipes)].drop_duplicates(subset=['recipe_id'])

    # replace NaN with None
    top_n_recipe_data = top_n_recipe_data.where(pd.notnull(top_n_recipe_data), None)
    top_n_recipe_data = top_n_recipe_data.head(n)
    top_n_recipe_data = top_n_recipe_data.to_dict(orient='records')
    return top_n_recipe_data
