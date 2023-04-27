import numpy as np
import tensorflow as tf


def nmf(X, latent_features, max_iter=100, error_limit=1e-6, fit_error_limit=1e-6):
    """
    Non-negative matrix factorization using TensorFlow 2.

    Args:
        X (ndarray): Input matrix of shape (n_samples, n_features).
        latent_features (int): Number of latent features.
        max_iter (int): Maximum number of iterations.
        error_limit (float): Error limit for convergence.
        fit_error_limit (float): Fit error limit for convergence.

    Returns:
        tuple: Tuple of matrices (W, H) such that X = W @ H, where W has shape (n_samples, latent_features) and H has shape (latent_features, n_features).
    """
    # Initialize matrices W and H
    n_samples, n_features = X.shape
    initializer = tf.random_normal_initializer()
    W = tf.Variable(initializer([n_samples, latent_features]), dtype=tf.float32)
    H = tf.Variable(initializer([latent_features, n_features]), dtype=tf.float32)

    # Define loss function
    X_tf = tf.constant(X, dtype=tf.float32)

    def loss_fn():
        return tf.reduce_sum(tf.square(X_tf - tf.matmul(W, H)))

    # Train model
    prev_error = 0
    for i in range(max_iter):
        with tf.GradientTape() as tape:
            loss = loss_fn()
        gradients = tape.gradient(loss, [W, H])

        optimizer = tf.keras.optimizers.legacy.Adam()
        optimizer.apply_gradients(zip(gradients, [W, H]))

        error = loss_fn().numpy()

        # Check for convergence
        if i % 10 == 0:
            print("Iteration {}: error = {}".format(i, error))
        if abs(prev_error - error) < error_limit:
            print("Converged to error = {}".format(error))
            break
        if error < fit_error_limit:
            print("Reached fit error limit of {}".format(fit_error_limit))
            break
        prev_error = error

    return W.numpy(), H.numpy()


# Create a new matrix of size 10000 x 500 with random values
X = np.random.rand(1000000, 500)

W, H = nmf(X, latent_features=2)
print("W =\n", W)
print("H =\n", H)

# Find the top 5 similar rows to the first row
first_row = X[0]
similar_rows = X[np.argsort(-W[:, 0])[:5]]
print("The top 5 similar rows to the first row are:\n", similar_rows)