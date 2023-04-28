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
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF

# create a random matrix of size 1000 x 50 which contains 60% zeros
X = np.random.rand(1000000, 50)

X[X < 0.6] = 0

X_sparse = csr_matrix(X)

model = NMF(n_components=100, init='random', random_state=0, max_iter=5000, solver='mu', l1_ratio=0.5, verbose=3)

W = model.fit_transform(X_sparse)
H = model.components_

loss = model.reconstruction_err_
print("loss: ", loss)
print("X: ", X)
print("W @ H: ", W @ H)

# save the model

with open('nmf_model.pkl', 'wb') as f:
    pickle.dump(model, f)
