"""The model here is heavily inspired by https://github.com/jaesik817/SequentialData-GAN/blob/master/gan.py
    Finding well structured GAN models online was very difficult but the way this model is built makes sense."""

import tensorflow as tf
import feed
import numpy as np
import os
import csv2midi
from datetime import datetime
from sklearn.model_selection import train_test_split
from csv import writer, reader

batches = feed.Scavenge()
X_train = batches
n_steps = 40
n_inputs = 2
n_neurons = 500
n_outputs = 2
n_output_steps = 40
n_layers = 8
batch_size = 10
eta = 0.0003

#Shuffles and returns the batches 
def get_batches(batch_size):
    #randomly generate a list of numbers to choose from for building the batch.
    starting = np.random.permutation(len(X_train))
    batch_test = starting[:batch_size]
    batch_list = np.asarray([X_train[x] for x in batch_test])
    if(batch_size == 1):
        #Hacky way to return a certain formated output for seeding the the network.
        return batch_list[:, :n_steps].reshape(-1, n_steps, n_inputs)
    return batch_list[:, :n_steps].reshape(-1, n_steps, n_inputs), batch_list[:, 1:].reshape(-1, n_steps, n_outputs)


def generator(X):
    with tf.variable_scope("GAN/Generator"):
        gru_cell = [tf.nn.rnn_cell.GRUCell(n_neurons) for layer in range (n_layers)]
        multi_cell = tf.nn.rnn_cell.MultiRNNCell(gru_cell)
        rnn_outputs, states = tf.nn.dynamic_rnn(multi_cell, X, dtype=tf.float32)

        #Takes the rnn outputs and stacks them to reduce dimensionality 
        stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, n_neurons])
        stacked_outputs = tf.layers.dense(stacked_rnn_outputs, n_outputs)

        #The outputs of the stacked layers.
        outputs = tf.reshape(stacked_outputs, [-1, n_output_steps, n_outputs])

        return outputs

def discriminator(X,reuse=False):
    with tf.variable_scope("GAN/Discriminator",reuse=reuse):
        gru_cell = [tf.nn.rnn_cell.GRUCell(n_neurons) for layer in range (n_layers)]
        multi_cell = tf.nn.rnn_cell.MultiRNNCell(gru_cell)
        rnn_outputs, states = tf.nn.dynamic_rnn(multi_cell, X, dtype=tf.float32)

        #Takes the rnn outputs and stacks them to reduce dimensionality 
        stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, n_neurons])
        stacked_outputs = tf.layers.dense(stacked_rnn_outputs, n_outputs)

        #The outputs of the stacked layers.
        outputs = tf.reshape(stacked_outputs, [-1, n_output_steps, n_outputs])
        logits = tf.layers.dense(outputs, 1)

        return logits


X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder (tf.float32, [None, n_steps, n_outputs])
G_sample = generator(X)
r_logits = discriminator(y)
f_logits = discriminator(G_sample,reuse=True)
disc_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=r_logits,labels=tf.ones_like(r_logits)) + tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logits,labels=tf.zeros_like(f_logits)))
gen_loss = tf.reduce_mean(tf.square(G_sample - y))

gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="GAN/Generator")
disc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="GAN/Discriminator")

gen_step = tf.train.AdamOptimizer(learning_rate=eta).minimize(gen_loss)
disc_step = tf.train.AdamOptimizer(learning_rate=eta).minimize(disc_loss)

n_epochs = 10000000
batch_size = 10
saver = tf.train.Saver()
best_path = "model/best_model"
current_path = "model/current_path"

#Takes the output of the network and converts it to midi.
def Output(epoch, current_path, sess):
    saver.restore(sess, current_path)
    seed = get_batches(1)
    X_batch = np.array(seed).reshape(1, n_steps, n_outputs)
    y_pred = sess.run(G_sample, feed_dict={X: X_batch})
    for line in y_pred[0]:
        if(os.path.isfile("outputs/" + str(epoch) + "_" + now + ".csv")):
            with open("outputs/" + str(epoch) + "_" + now + ".csv", "a") as output_file:
                output_file.write(str(round(line[0])) + "," + str(round(line[1])) + "\n")
        else:    
            with open("outputs/" + str(epoch) + "_" + now +".csv", "w+") as output_file:
                output_file.write(str(round(line[0])) + "," + str(round(line[1])) + "\n")
    #Convert to midi
    csv2midi.Work("outputs/" + str(epoch) + "_" + now + ".csv")

init = tf.global_variables_initializer()
with tf.Session() as sess:
    for epoch in range(n_epochs):
        init.run(session=sess)
        best_mse = np.infty
        X_batch, y_batch = get_batches(batch_size)
        _, dloss = sess.run([disc_step, disc_loss], feed_dict={X: X_batch, y: y_batch})
        _, gloss = sess.run([gen_step, gen_loss], feed_dict={y: y_batch})
        if epoch % 100 ==0:
            print ("Iterations: {}\t Discriminator loss: {}\t Generator loss: {}".format(epoch,dloss,gloss))
            saver.save(sess, current_path)
            Output(epoch, current_path, sess)
            mse = loss.eval(feed_dict={X: X_batch, y: y_batch})
            print(epoch, "MSE:", mse)
            if mse < best_mse:
                best_mse = mse
                print("New best model")
                sample = outputs.eval(feed_dict={X: X_batch, y: y_batch})
                print("sample:", sample[0] )
                saver.save(sess, best_path)