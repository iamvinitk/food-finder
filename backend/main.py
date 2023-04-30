import pandas as pd
from flask import Flask, request

from cdl import get_similar_recipes_cdl
from hybrid import get_similar_recipes_hybrid
from nmf import get_nmf_recommendations

app = Flask(__name__)

df = pd.read_csv('../dataset/preprocessed_data.csv')

search_df = df.copy(deep=True)
search_df = search_df[['recipe_id', 'name']]
# drop duplicates
search_df = search_df.drop_duplicates(subset=['recipe_id'])

search_df['recipe_count'] = df.groupby(['recipe_id'])['recipe_id'].transform('count')


@app.route("/model/<model_name>")
def hello_world(model_name):
    recipe_id = request.args.get('recipe_id')
    recipe_id = int(recipe_id)
    if model_name == "hybrid":
        recommendations = get_similar_recipes_hybrid(df, recipe_id=recipe_id)
        return recommendations

    if model_name == "nmf":
        recommendations = get_nmf_recommendations(df, recipe_id=recipe_id)
        return recommendations

    if model_name == "cdl":
        recommendations = get_similar_recipes_cdl(df, recipe_id=recipe_id)
        return recommendations
    return f'Hello, World! {model_name} {recipe_id}'


@app.route("/search")
def search():
    query = request.args.get('query')
    query = query.lower()
    results = search_df[search_df['name'].str.contains(query)]
    results = results.sort_values(by=['recipe_count'], ascending=False)
    results = results.head(10)
    return results.to_dict(orient='records')
