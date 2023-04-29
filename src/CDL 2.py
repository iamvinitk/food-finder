import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model

data = pd.read_csv('../dataset/preprocessed_data.csv')

df = data[['ingredients', 'recipe_id']]
df.head(3)

# find the count of duplicate rows
df.duplicated().sum()

# drop duplicate rows
df.drop_duplicates(inplace=True)

df['ingredients'] = df['ingredients'].apply(lambda x: x.replace(', ', ' '))
df.head(3)

# Convert ingredients to numerical vectors
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(df['ingredients'])

ingredients = tokenizer.texts_to_sequences(df['ingredients'])
max_length = max([len(seq) for seq in ingredients])

# Pad the sequences to the max length
ingredients = tf.keras.preprocessing.sequence.pad_sequences(ingredients, maxlen=max_length)
ingredients.shape

# Convert recipe_id to numerical vectors
label_encoder = LabelEncoder()
recipe_ids = label_encoder.fit_transform(df['recipe_id'])

train_ingredients, test_ingredients, train_recipe_ids, test_recipe_ids = train_test_split(ingredients, recipe_ids,
                                                                                          test_size=0.2)

# Define the stacked de-noising autoencoder
input_layer = Input(shape=(max_length,))
x = Dense(256, activation='relu')(input_layer)
x = Dense(128, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(32, activation='relu')(x)
encoded = Dense(16, activation='relu')(x)
x = Dense(32, activation='relu')(encoded)
x = Dense(64, activation='relu')(x)
x = Dense(128, activation='relu')(x)
x = Dense(256, activation='relu')(x)
decoded = Dense(max_length, activation='linear')(x)

autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001), loss='mse')

# Train the stacked de-noising autoencoder
autoencoder.fit(train_ingredients, train_ingredients, epochs=5, batch_size=32,
                validation_data=(test_ingredients, test_ingredients))

# Fine-tune the autoencoder with a supervised learning model
autoencoder.layers

# Get the encoder layers from the autoencoder
encoder_layers = autoencoder.layers[:5]
encoder_layers

# Define the classifier on top of the encoder layers
input_layer = encoder_layers[0].input
x = encoder_layers[-1].output
x = Dense(64, activation='relu')(x)
x = Dropout(0.5)(x)
output_layer = Dense(1, activation='sigmoid')(x)

# Create a new model with the encoder layers and the classifier
model = Model(inputs=input_layer, outputs=output_layer)

# Compile the model with binary_crossentropy loss and accuracy metric
model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001), loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train_ingredients, train_recipe_ids, epochs=100, batch_size=32,
          validation_data=(test_ingredients, test_recipe_ids))
