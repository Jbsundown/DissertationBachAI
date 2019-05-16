from functools import partial
from sklearn.model_selection import train_test_split
import tensorflow as tf
import feed

data = feed.Scavenge()
X_train, X_test = train_test_split(data,test_size=0.1, random_state=26)

epochs = 50

n_inputs = 3
n_hidden_1 = 2
h_mid_neurons = 1
n_hidden_2 = 2
n_outputs = n_inputs
eta = 0.01

initialiser = tf.contrib.layers.variance_scaling_initializer()
dense_layer=partial(tf.layers.dense, activation=tf.nn.elu, kernel_initializer=initialiser)

X = tf.placeholder(tf.float32, [None, n_inputs])
h1 = dense_layer(X, n_hidden_1)
h_mid_mean = dense_layer(h1, h_mid_neurons, activation=None)
h_mid_gamma = dense_layer(h1, h_mid_neurons, activation=None)
noise = tf.random_normal(tf.shape(h_mid_gamma), dtype=tf.float32)
h_mid = h_mid_mean + tf.exp(0.5 * h_mid_gamma) * noise
h2 = dense_layer(h_mid, n_hidden_2)
logits = dense_layer(h2, n_outputs, activation=None)
outputs = tf.sigmoid(logits)
print("Graph built")

cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=X, logits=logits)
reconstruction_loss = tf.reduce_sum(cross_entropy)
latent_loss = 0.5 * tf.reduce_sum(tf.exp(h_mid_gamma) + tf.square(h_mid_mean) -1 - h_mid_gamma)
loss = reconstruction_loss + latent_loss

optimiser = tf.train.AdamOptimizer(eta)
training_op = optimiser.minimize(loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

codings = h_mid
with tf.Session() as sess:
    init.run()
    print("Running AE")
    for epoch in range(epochs):
        training_op.run(feed_dict={X:X_train})
        if epoch % 10 == 0:
            print(epoch, ": Loss:", loss.eval(feed_dict={X:X_test}))
    codings_val = codings.eval(feed_dict={X:X_test})
    results = logits.eval(feed_dict={X:X_test})
    print(results)
    print(codings_val)