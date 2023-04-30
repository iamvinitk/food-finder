import numpy as np

W = np.load('./saved_models/nmf_W.npy')
H = np.load('./saved_models/nmf_H.npy')


def get_nmf_recommendations(dataset, recipe_id):
    # get the index of the recipe id
    recipe_id = int(recipe_id)
    print(dataset[dataset['recipe_id'] == recipe_id].index)
    recipe_index = dataset[dataset['recipe_id'] == recipe_id].index[0]

    # get the factor values of the recipe
    recipe_factors = W[recipe_index]

    # calculate the similarity of this recipe with others
    similarity = np.dot(W, recipe_factors)

    # get the indices of the top 10 most similar recipes (excluding itself)
    top10_indices = np.argsort(similarity)[-101:-1]

    # print the top 10 most similar recipes
    x = dataset.iloc[top10_indices]

    # remove the recipe itself and repeated recipe_ids
    x = x[x['recipe_id'] != recipe_id]
    x = x.drop_duplicates(subset=['recipe_id'])

    # select the top 10 most similar recipes
    x = x.head(10)
    return x.to_dict(orient='records')


def get_nmf_user_recommendations(dataset, user_id):
    # get the index of the user id
    user_id = int(user_id)
    user_index = dataset[dataset['user_id'] == user_id].index[0]

    # get the factor values of the user
    user_factors = H[:, user_index]

    # calculate the similarity of this user with other users
    similarity = np.dot(H, user_factors)

    # get the indices of the top 10 most similar users (excluding itself)
    top10_indices = np.argsort(similarity)[-11:-1]

    # get the recipe ids of the top 10 most similar users
    top10_recipe_ids = dataset.iloc[top10_indices]['recipe_id'].tolist()

    # get the indices of the top 10 most similar recipes (excluding itself)
    top10_indices = np.argsort(similarity)[-101:-1]

    # get the recipe ids of the top 10 most similar recipes
    x = dataset.iloc[top10_indices]

    # remove the recipe itself and repeated recipe_ids
    x = x[x['recipe_id'].isin(top10_recipe_ids)]
    x = x.drop_duplicates(subset=['recipe_id'])

    # select the top 10 most similar recipes
    top10_recipes = x.head(10)

    return top10_recipes.to_dict(orient='records')
