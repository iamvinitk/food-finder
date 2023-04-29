import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

ratings = tfds.load("movielens/100k-ratings", split="train")
movie_title_lookup = tf.keras.layers.StringLookup()

movie_title_lookup.adapt(ratings.map(lambda x: x["movie_title"]))

num_hashing_bins = 200_000

movie_title_hashing = tf.keras.layers.Hashing(
    num_bins=num_hashing_bins
)

movie_title_embedding = tf.keras.layers.Embedding(
    # Let's use the explicit vocabulary lookup.
    input_dim=movie_title_lookup.vocab_size(),
    output_dim=32
)

movie_title_model = tf.keras.Sequential([movie_title_lookup, movie_title_embedding])

user_id_lookup = tf.keras.layers.StringLookup()
user_id_lookup.adapt(ratings.map(lambda x: x["user_id"]))

user_id_embedding = tf.keras.layers.Embedding(user_id_lookup.vocab_size(), 32)

user_id_model = tf.keras.Sequential([user_id_lookup, user_id_embedding])

timestamp_normalization = tf.keras.layers.Normalization(
    axis=None
)
timestamp_normalization.adapt(ratings.map(lambda x: x["timestamp"]).batch(1024))

max_timestamp = ratings.map(lambda x: x["timestamp"]).reduce(
    tf.cast(0, tf.int64), tf.maximum).numpy().max()
min_timestamp = ratings.map(lambda x: x["timestamp"]).reduce(
    np.int64(1e9), tf.minimum).numpy().min()

timestamp_buckets = np.linspace(
    min_timestamp, max_timestamp, num=1000)

timestamp_embedding_model = tf.keras.Sequential([
    tf.keras.layers.Discretization(timestamp_buckets.tolist()),
    tf.keras.layers.Embedding(len(timestamp_buckets) + 1, 32)
])

title_text = tf.keras.layers.TextVectorization()
title_text.adapt(ratings.map(lambda x: x["movie_title"]))


class UserModel(tf.keras.Model):

    def __init__(self):
        super().__init__()

        self.user_embedding = tf.keras.Sequential([
            user_id_lookup,
            tf.keras.layers.Embedding(user_id_lookup.vocab_size(), 32),
        ])
        self.timestamp_embedding = tf.keras.Sequential([
            tf.keras.layers.Discretization(timestamp_buckets.tolist()),
            tf.keras.layers.Embedding(len(timestamp_buckets) + 2, 32)
        ])
        self.normalized_timestamp = tf.keras.layers.Normalization(
            axis=None
        )

    def call(self, inputs):
        # Take the input dictionary, pass it through each input layer,
        # and concatenate the result.
        return tf.concat([
            self.user_embedding(inputs["user_id"]),
            self.timestamp_embedding(inputs["timestamp"]),
            tf.reshape(self.normalized_timestamp(inputs["timestamp"]), (-1, 1))
        ], axis=1)


user_model = UserModel()

user_model.normalized_timestamp.adapt(
    ratings.map(lambda x: x["timestamp"]).batch(128))


class MovieModel(tf.keras.Model):

    def __init__(self):
        super().__init__()

        max_tokens = 10_000

        self.title_embedding = tf.keras.Sequential([
            movie_title_lookup,
            tf.keras.layers.Embedding(movie_title_lookup.vocab_size(), 32)
        ])
        self.title_text_embedding = tf.keras.Sequential([
            tf.keras.layers.TextVectorization(max_tokens=max_tokens),
            tf.keras.layers.Embedding(max_tokens, 32, mask_zero=True),
            tf.keras.layers.GlobalAveragePooling1D(),
        ])

    def call(self, inputs):
        return tf.concat([
            self.title_embedding(inputs["movie_title"]),
            self.title_text_embedding(inputs["movie_title"]),
        ], axis=1)
