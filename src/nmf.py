# import numpy as np
# from scipy.sparse import csr_matrix
# from sklearn.decomposition import NMF
#
# # Load your data into a dense matrix (X)
# X = np.random.rand(1000, 50)
#
# # Convert the matrix to a sparse matrix
# X_sparse = csr_matrix(X)
#
# # Create an instance of the NMF object with the randomized solver
# nmf = NMF(n_components=10, max_iter=1000000000)
#
# # Use the batch_size parameter to break the computation into smaller batches
# nmf.batch_size = 1000
#
# # Use parallel processing to speed up the computation
# nmf.n_jobs = -1
#
# # Fit the model to the data
# nmf.fit(X_sparse)
#
# # Extract the latent factors
# W = nmf.transform(X_sparse)
#
# # Extract the feature weights
# H = nmf.components_
#
# # Print the loss value
# print(nmf.reconstruction_err_)
#
# # print X
# print("X: ", X)
#
# # print W @ H
# print(W @ H)

import pickle

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF

data = pd.read_csv('../dataset/temp.csv')

X = data.drop(['recipe_id'], axis=1)

# find the values in X that are smaller than 0
print(X.shape)
print(X.head(4))

n_components = 2

# create an NMF object and fit the data
model = NMF(n_components=n_components, init='random', random_state=0)
model.fit(X)

# get the factorization matrices
W = model.transform(X)
H = model.components_

# print the results
print("Factorization matrix W:")
print(W)
print("Factorization matrix H:")
print(H)

loss = model.reconstruction_err_
print("loss: ", loss)
print("X: ", X.values)
print("W @ H: ", W @ H)

# save the model

with open('nmf_model.pkl', 'wb') as f:
    pickle.dump(model, f)

row_index = 0  # for example
selected_row = W[row_index, :]

similarities = W.dot(selected_row)
similar_indices = np.argsort(-similarities)[:10]
similar_rows = data.iloc[similar_indices, :]

print("similar_rows: ", similar_rows)