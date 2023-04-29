import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from gensim import corpora, models
import os
# print pwd
# print os.getcwd()
print(os.getcwd())

# Load the dataset into a pandas dataframe
data = pd.read_csv('/Users/sanikakarwa/Desktop/CMPE-256-Food-Recommendation/dataset/RAW_recipes.csv')

# Combine the recipe descriptions, ingredients, and review texts into a single column
data['text'] = data[['description', 'ingredients']].apply(lambda x: ', '.join(x), axis=1)
data['text'] = data[['text', 'minutes', 'n_steps']].apply(lambda x: ', '.join(x), axis=1)

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words='english')

# Vectorize the text data
X = vectorizer.fit_transform(data['text'])

# Perform NMF to extract topics
nmf = NMF(n_components=10, random_state=42)
nmf.fit(X)

# Get the top words for each topic
feature_names = vectorizer.get_feature_names()
for topic_idx, topic in enumerate(nmf.components_):
    print(f"Topic {topic_idx}:")
    print(", ".join([feature_names[i] for i in topic.argsort()[:-11:-1]]))
    print()

# Create a dictionary and a corpus for LDA
docs = [doc.split(", ") for doc in data['text']]
dictionary = corpora.Dictionary(docs)
corpus = [dictionary.doc2bow(doc) for doc in docs]

# Train an LDA model with 10 topics
lda = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=10, passes=10)

# Print the topics and their top words
for i, topic in lda.show_topics(formatted=True, num_topics=10, num_words=10):
    print(f"Topic {i}: {topic}")

# Make recommendations based on the topics
topic_probs = lda.get_document_topics(corpus)
recommendations = []
for i, probs in enumerate(topic_probs):
    topic = max(probs, key=lambda x: x[1])[0]
    recipes = data[data.index != i][data['text'].apply(lambda x: topic in x)]
    if not recipes.empty:
        recommendations.append((data.iloc[i]['recipe_id'], recipes['recipe_id'].tolist()))

print(recommendations)
