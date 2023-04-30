import pandas as pd
from flask import Flask, request

from hybrid import get_similar_recipes_hybrid
from nmf import get_nmf_recommendations

app = Flask(__name__)

df = pd.read_csv('../dataset/preprocessed_data.csv')


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
    return f'Hello, World! {model_name} {recipe_id}'
