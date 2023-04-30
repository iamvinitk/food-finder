import json

import faiss
import numpy as np
import pandas as pd
from tensorflow import keras

model = keras.models.load_model('recommendation')

recipe2recipe_encoded = json.load(open('recommendation/recipe2recipe_encoded.json'))

print(recipe2recipe_encoded)

recipe_embedding = model.recipe_embedding.get_weights()[0]
print(recipe_embedding.shape)

index = faiss.IndexFlatL2(recipe_embedding.shape[1])
index.add(recipe_embedding)


def get_similar_recipes_cdl(df, recipe_id):
    recipe_index = recipe2recipe_encoded[str(recipe_id)]
    recipe_vector = recipe_embedding[recipe_index]
    # search for the nearest neighbors
    k = 50
    D, I = index.search(np.array([recipe_vector]), k)
    recipe_ids = [int(i) for i in I[0]]

    decoded_recipe_ids = [list(recipe2recipe_encoded.keys())[list(recipe2recipe_encoded.values()).index(i)] for i in
                          recipe_ids]
    decoded_recipe_ids = [int(i) for i in decoded_recipe_ids]
    top_n = df[df.recipe_id.isin(decoded_recipe_ids)].drop_duplicates(subset=['recipe_id'])
    top_n = top_n.where(pd.notnull(top_n), None)
    top_n = top_n.head(10)
    return top_n.to_dict(orient='records')
