import json
import pickle

import pandas as pd

filename = 'saved_models/svd_model.sav'
algo = pickle.load(open(filename, 'rb'))


def get_svd_recommendations(df, user_id, n=10):
    user_ratings = df[df.user_id == user_id]
    rated_recipes = user_ratings.recipe_id.unique().tolist()
    unrated_recipes = df[~df.recipe_id.isin(rated_recipes)].recipe_id.unique().tolist()

    test_set = [[user_id, recipe_id, 0] for recipe_id in unrated_recipes]
    predictions = algo.test(test_set)

    # sort predictions by estimated rating
    predictions.sort(key=lambda x: x.est, reverse=True)

    top_n = predictions[:n]
    top_n_recipes = [recipe_id for user_id, recipe_id, _, _, _ in top_n]
    # return the recipe_id from df that match the top_n_recipes
    top_n_recipe_data = df[df.recipe_id.isin(top_n_recipes)].drop_duplicates(subset=['recipe_id'])

    # convert to json
    top_n_recipe_data = top_n_recipe_data.to_dict(orient='records')
    return top_n_recipe_data


def get_svd_similar_recipes(df, recipe_id, n=10):
    recipe_ratings = df[df.recipe_id == recipe_id]
    recipe_ratings['rating'] = recipe_ratings['rating'].astype(float)
    recipe_mean_rating = recipe_ratings['rating'].mean()

    test_set = [[user_id, recipe_id, recipe_mean_rating] for user_id in df.user_id.unique()]
    predictions = algo.test(test_set)

    # filter predictions for the target recipe
    target_predictions = [pred for pred in predictions if pred.iid == recipe_id]

    # sort target predictions by estimated rating
    target_predictions.sort(key=lambda x: x.est, reverse=True)

    top_n = target_predictions[:n]
    top_n_recipes = [pred.uid for pred in top_n]

    # return the recipe_id from df that match the top_n_recipes
    top_n_recipe_data = df[df.user_id.isin(top_n_recipes)].drop_duplicates(subset=['recipe_id'])

    # convert to json
    # replace NaN with None
    top_n_recipe_data = top_n_recipe_data.where(pd.notnull(top_n_recipe_data), None)
    top_n_recipe_data = top_n_recipe_data.head(n)
    top_n_recipe_data = top_n_recipe_data.to_dict(orient='records')
    return top_n_recipe_data
