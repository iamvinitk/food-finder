import pandas as pd
import tensorflow as tf
import tensorflow_recommenders as tfrs

# Load the dataset into a pandas dataframe
data = pd.read_csv("../dataset/preprocessed_data.csv")

data = data[["user_id", "recipe_id", "rating", "total_fat", "sugar", "sodium", "protein", "saturated_fat",
             "carbohydrates"]]

data["user_id"] = data["user_id"].astype(str)
data["recipe_id"] = data["recipe_id"].astype(str)

# Create user and item datasets
user_df = data[["user_id"]]
item_df = data[["recipe_id"]]

# Normalize continuous features
data["calories"] = (data["calories"] - data["calories"].mean()) / data["calories"].std()
data["total_fat"] = (data["total_fat"] - data["total_fat"].mean()) / data["total_fat"].std()
data["sugar"] = (data["sugar"] - data["sugar"].mean()) / data["sugar"].std()
data["sodium"] = (data["sodium"] - data["sodium"].mean()) / data["sodium"].std()
data["protein"] = (data["protein"] - data["protein"].mean()) / data["protein"].std()
data["saturated_fat"] = (data["saturated_fat"] - data["saturated_fat"].mean()) / data["saturated_fat"].std()
data["carbohydrates"] = (data["carbohydrates"] - data["carbohydrates"].mean()) / data["carbohydrates"].std()

# Convert data to TensorFlow Datasets
tf_data = tf.data.Dataset.from_tensor_slices(dict(data))

# Split data into training and testing sets
tf_train_data = tf_data.take(int(len(data) * 0.8)).shuffle(10000).batch(256)
tf_test_data = tf_data.skip(int(len(data) * 0.8)).batch(256)

# Define user and item embeddings
user_embeddings = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.StringLookup(
        vocabulary=user_df["user_id"].unique(), mask_token=None),
    tf.keras.layers.Embedding(len(user_df["user_id"].unique()) + 1, 32)
])
item_embeddings = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.StringLookup(
        vocabulary=item_df["recipe_id"].unique(), mask_token=None),
    tf.keras.layers.Embedding(len(item_df["recipe_id"].unique()) + 1, 32)
])


# Define model
class RecipeModel(tfrs.Model):

    def __init__(self, user_embeddings, item_embeddings):
        super().__init__()
        self.user_embeddings = user_embeddings
        self.item_embeddings = item_embeddings
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=item_df["recipe_id"].unique().tolist())
        )

    def compute_loss(self, features, training=False):
        user_embeddings = self.user_embeddings(features["user_id"])
        positive_embeddings = self.item_embeddings(features["recipe_id"])
        return self.task(user_embeddings, positive_embeddings)


# Instantiate model
model = RecipeModel(user_embeddings, item_embeddings)
model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))

# Train model
model.fit(tf_train_data, epochs=10)

# Evaluate model
model.evaluate(tf_test_data, return_dict=True)

# Generate recommendations for a given user
user_id = "492"
user_embedding = model.user_embeddings(tf.constant(user_id))
scores = tf.matmul(user_embedding, tf.transpose(model.item_embeddings.weights[0]))
top_recipe_indices = tf.argsort(scores, direction="DESCENDING")
top_recipe_ids = [item_df["recipe_id"].unique()[i] for i in top_recipe_indices.numpy()[0][:10]]
print("Top recommendedrecipes for user", user_id, ":", top_recipe_ids)
# Save model
model.save("recipe_recommendation_model")

# Load model
loaded_model = tf.keras.models.load_model("recipe_recommendation_model")

# Generate recommendations using loaded model
user_id = "492"
user_embedding = loaded_model.user_embeddings(tf.constant(user_id))
scores = tf.matmul(user_embedding, tf.transpose(loaded_model.item_embeddings.weights[0]))
top_recipe_indices = tf.argsort(scores, direction="DESCENDING")
top_recipe_ids = [item_df["recipe_id"].unique()[i] for i in top_recipe_indices.numpy()[0][:10]]
print("Top recommended recipes for user", user_id, "using loaded model:", top_recipe_ids)
