import pandas as pd
from surprise import Reader, Dataset, SVD

df = pd.read_csv('../dataset/preprocessed_interactions.csv')

df.head()

reader = Reader()
data = Dataset.load_from_df(df[['user_id', 'recipe_id', 'rating']], reader)

trainset = data.build_full_trainset()

algo = SVD()

algo.fit(trainset)


def get_recommendations(user_id, n=10):
    user_ratings = df[df.user_id == user_id]
    rated_recipes = user_ratings.recipe_id.unique().tolist()
    unrated_recipes = df[~df.recipe_id.isin(rated_recipes)].recipe_id.unique().tolist()

    testset = [[user_id, recipe_id, 0] for recipe_id in unrated_recipes]
    predictions = algo.test(testset)

    # sort predictions by estimated rating
    predictions.sort(key=lambda x: x.est, reverse=True)

    top_n = predictions[:n]

    top_n_recipes = [recipe_id for user_id, recipe_id, _, _, _ in top_n]

    # return the recipe_id from df that match the top_n_recipes
    return df[df.recipe_id.isin(top_n_recipes)].drop_duplicates(subset=['recipe_id'])


print(get_recommendations(8937))
