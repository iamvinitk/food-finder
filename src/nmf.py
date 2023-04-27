# import numpy as np
# import tensorflow as tf
#
#
# def nmf(X, latent_features, max_iter=100, error_limit=1e-6, fit_error_limit=1e-6):
#     """
#     Non-negative matrix factorization using TensorFlow 2.
#
#     Args:
#         X (ndarray): Input matrix of shape (n_samples, n_features).
#         latent_features (int): Number of latent features.
#         max_iter (int): Maximum number of iterations.
#         error_limit (float): Error limit for convergence.
#         fit_error_limit (float): Fit error limit for convergence.
#
#     Returns:
#         tuple: Tuple of matrices (W, H) such that X = W @ H, where W has shape (n_samples, latent_features) and H has shape (latent_features, n_features).
#     """
#     # Initialize matrices W and H
#     n_samples, n_features = X.shape
#     initializer = tf.random_normal_initializer()
#     W = tf.Variable(initializer([n_samples, latent_features]), dtype=tf.float32)
#     H = tf.Variable(initializer([latent_features, n_features]), dtype=tf.float32)
#
#     # Define loss function
#     X_tf = tf.constant(X, dtype=tf.float32)
#
#     def loss_fn():
#         WH = tf.matmul(W, H)
#         diff = X_tf - WH
#         return tf.reduce_sum(tf.square(diff))
#
#     # Train model
#     optimizer = tf.optimizers.Adam()
#     prev_error = 0
#     for i in range(max_iter):
#         with tf.GradientTape() as tape:
#             loss = loss_fn()
#         gradients = tape.gradient(loss, [W, H])
#         optimizer.apply_gradients(zip(gradients, [W, H]))
#
#         error = loss_fn().numpy()
#
#         # Check for convergence
#         if i % 10 == 0:
#             print("Iteration {}: error = {}".format(i, error))
#         if abs(prev_error - error) < error_limit:
#             print("Converged to error = {}".format(error))
#             break
#         if error < fit_error_limit:
#             print("Reached fit error limit of {}".format(fit_error_limit))
#             break
#         prev_error = error
#
#     return W.numpy(), H.numpy()
#
#
# # Create a new matrix of size 10000 x 500 with random values
# # X = np.random.rand(1000000, 500)
#
# X = np.array([[1, 1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])
#
# W, H = nmf(X, latent_features=2, max_iter=5000)
# print("W =\n", W)
# print("H =\n", H)
#
# print(np.matmul(W, H))

# import numpy as np
# import tensorflow as tf
#
# # Define the input matrix X
# # X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# X = np.random.rand(10000, 50)
#
# # Define the dimensions of the matrices
# n, m = X.shape
# k = 2  # The number of components
#
# # Initialize the W and H matrices randomly
# W = tf.Variable(tf.random.normal((n, k), stddev=0.01))
# H = tf.Variable(tf.random.normal((k, m), stddev=0.01))
#
# # Set the learning rate and number of iterations
# learning_rate = 0.001
# num_iterations = 100000
#
#
# # Define the loss function as the Frobenius norm of the difference between X and W @ H
# def loss_fn(X, W, H):
#     return tf.reduce_sum(tf.pow(X - tf.matmul(W, H), 2))
#
#
# # Define the optimizer to minimize the loss
# optimizer = tf.keras.optimizers.legacy.SGD(learning_rate)
#
# # Train the model
# for i in range(num_iterations):
#     with tf.GradientTape() as tape:
#         loss = loss_fn(X, W, H)
#     gradients = tape.gradient(loss, [W, H])
#     optimizer.apply_gradients(zip(gradients, [W, H]))
#
# # Print the resulting matrices
# print("W:", W.numpy())
# print("H:", H.numpy())
#
# print("X:", X)
# print("W @ H:", tf.matmul(W, H).numpy())
#
# # Find the top 10 similar rows to the first row
# first_row = tf.reshape(X[0], (1, -1))
# distances = tf.norm(X - first_row, axis=1)
# top10 = tf.argsort(distances)[:10]
# print("Top 10 similar rows to the first row:", top10.numpy())


import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

X = np.random.rand(100, 5)

# Set the dimensions of the input matrix X and the number of latent features k
n, m = X.shape
k = 10

# Define the NMF model architecture
input_layer = layers.Input(shape=(m,))
dense_layer_1 = layers.Dense(units=k, activation="relu", use_bias=False)(input_layer)
dense_layer_2 = layers.Dense(units=m, activation="relu", use_bias=False)(dense_layer_1)


# Define the loss function
def nmf_loss(X, W, H):
    # Compute the Frobenius norm of the difference between X and WH
    diff = X - tf.matmul(W, H)
    frob_norm = tf.norm(diff, ord="fro", axis=[-2, -1])

    # Compute the Frobenius norm of X
    X_norm = tf.norm(X, ord="fro", axis=[-2, -1])

    # Compute the NMF loss
    nmf_loss = frob_norm / X_norm

    return nmf_loss


# Define the optimizer
optimizer = optimizers.Adam(learning_rate=0.01)

# Define the NMF model
nmf_model = models.Model(inputs=input_layer, outputs=dense_layer_2)

# Compile the model
nmf_model.compile(optimizer=optimizer, loss=nmf_loss)

# Initialize W and H
W_init = np.random.rand(n, k)
H_init = np.random.rand(k, m)

# Train the model
nmf_model.fit(X, X, epochs=100, verbose=1)

# Extract the learned factor matrices W and H
W = nmf_model.layers[1].get_weights()[0]
H = nmf_model.layers[2].get_weights()[0]

# Print the resulting matrices
print("W:", W)
print("H:", H)
print("X:", X)
print("W @ H:", tf.matmul(W, H).numpy())
