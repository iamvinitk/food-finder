import numpy as np
import tensorflow as tf

# define the input data and parameters
num_recipes = 1000  # number of recipes in the dataset
num_attributes = 50  # number of recipe attributes
num_hidden = 100  # number of hidden units in the SDAE layers
num_latent = 10  # number of latent factors to learn
learning_rate = 0.01  # learning rate for optimization
num_epochs = 100  # number of training epochs

# generate random recipe attributes as input data
input_data = np.random.rand(num_recipes, num_attributes)

# define the SDAE layers for content modeling
input_layer = tf.keras.layers.Input(shape=(num_attributes,))
encoder_layer1 = tf.keras.layers.Dense(num_hidden, activation='relu')(input_layer)
encoder_layer2 = tf.keras.layers.Dense(num_latent, activation='relu')(encoder_layer1)
decoder_layer1 = tf.keras.layers.Dense(num_hidden, activation='relu')(encoder_layer2)
decoder_layer2 = tf.keras.layers.Dense(num_attributes, activation='sigmoid')(decoder_layer1)
sd_encoder = tf.keras.models.Model(input_layer, encoder_layer2)

# define the matrix factorization model for user-item interactions
user_input = tf.keras.layers.Input(shape=(1,), dtype='int32')
item_input = tf.keras.layers.Input(shape=(1,), dtype='int32')
embedding_layer = tf.keras.layers.Embedding(num_recipes, num_latent, input_length=1)
user_embedding = embedding_layer(user_input)
item_embedding = embedding_layer(item_input)
user_bias = tf.keras.layers.Embedding(num_recipes, 1, input_length=1)(user_input)
item_bias = tf.keras.layers.Embedding(num_recipes, 1, input_length=1)(item_input)
dot_product = tf.keras.layers.Dot(axes=2)([user_embedding, item_embedding])
merged = tf.keras.layers.Add()([dot_product, user_bias, item_bias])
mf_model = tf.keras.models.Model(inputs=[user_input, item_input], outputs=merged)

# combine the SDAE and MF models for recipe recommendation
input_user = tf.keras.layers.Input(shape=(1,), dtype='int32')
input_item = tf.keras.layers.Input(shape=(1,), dtype='int32')
content_embedding = sd_encoder(input_data)
mf_embedding_user = tf.keras.layers.Flatten()(embedding_layer(input_user))
mf_embedding_item = tf.keras.layers.Flatten()(embedding_layer(input_item))
mf_output = mf_model([input_user, input_item])
combined_output = tf.keras.layers.Concatenate()([content_embedding, mf_embedding_user, mf_embedding_item, mf_output])
recommendation_model = tf.keras.models.Model(inputs=[input_data, input_user, input_item], outputs=combined_output)

# compile the model and train on the dataset
recommendation_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
for epoch in range(num_epochs):
    history = recommendation_model.fit([input_data, np.random.randint(num_recipes, size=num_recipes),
                                        np.random.randint(num_recipes, size=num_recipes)],
                                       np.zeros((num_recipes, num_attributes + 2 * num_latent + 1)),
                                       batch_size=32, epochs=1, verbose=0)
    print('Epoch {}: loss={}'.format(epoch + 1, history.history['loss'][0]))


# generate recommendations for a given recipe
def recommend_similar_recipes(recipe_id, num_recommendations):
    # Construct input data for recipe, user, and item
    recipe_data = np.expand_dims(input_data[recipe_id], axis=0)
    user_id = np.ones((num_recipes, 1)) * recipe_id
    item_ids = np.arange(num_recipes)
    item_ids = np.delete(item_ids, recipe_id)
    item_embeddings = embedding_layer(item_ids)
    user_embedding = np.ones((num_recipes, 1)) * recipe_id
    user_bias = np.zeros((num_recipes, 1))
    item_bias = np.zeros((num_recipes, 1))
    mf_output = mf_model([user_embedding, item_ids.reshape((-1, 1))])
    combined_output = np.concatenate([recipe_data, item_embeddings, mf_output, user_bias, item_bias], axis=1)

    # Generate recommendations
    scores = recommendation_model.predict(combined_output, batch_size=32)
    sorted_indices = np.argsort(-scores.squeeze())[:num_recommendations]
    return sorted_indices.tolist()


# test the recommendation function for a random recipe
recipe_id = np.random.randint(num_recipes)
recommended_ids = recommend_similar_recipes(recipe_id, 10)
print('Recommended recipes for recipe {}: {}'.format(recipe_id, recommended_ids))
