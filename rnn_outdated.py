import tensorflow as tf
import feed
import numpy as np
import os
import csv2midi
from datetime import datetime
from sklearn.model_selection import train_test_split
from csv import writer, reader

X_train = feed.Scavenge()
n_steps = 50
n_inputs = 2
n_neurons = 600
n_outputs = 2
n_output_steps = 50
n_layers = 8
eta = 0.0001

#Shuffles and returns the batches 
def get_batches(batch_size):
    #randomly generate a list of numbers to choose from for building the batch.
    starting = np.random.permutation(len(X_train))
    batch_test = starting[:batch_size]
    batch_list = np.asarray([X_train[x] for x in batch_test])
    if(batch_size == 1):
        #Hacky way to return a certain formated output for seeding the network.
        return batch_list[:, :n_steps].reshape(-1, n_steps, n_inputs)
    return batch_list[:, :n_steps].reshape(-1, n_steps, n_inputs), batch_list[:, 1:].reshape(-1, n_steps, n_outputs)


#Starts building the graph of the network. The inputs and the cells they feed into.
X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder (tf.float32, [None, n_steps, n_outputs])
gru_cell = [tf.nn.rnn_cell.GRUCell(n_neurons) for layer in range (n_layers)]
multi_cell =  tf.nn.rnn_cell.MultiRNNCell(gru_cell)
rnn_outputs, states = tf.nn.dynamic_rnn(multi_cell, X, dtype=tf.float32)

#Takes the rnn outputs and stacks them to reduce dimensionality 
stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, n_neurons])
stacked_outputs = tf.layers.dense(stacked_rnn_outputs, n_outputs)

#The outputs of the stacked layers.
outputs = tf.reshape(stacked_outputs, [-1, n_output_steps, n_outputs])

#The loss, the number the model is trying to make as small as possible and the optimiser it's using to try to do that.
loss = tf.reduce_mean(tf.square(outputs - y))
optimiser = tf.train.AdamOptimizer(learning_rate=eta)
training_op = optimiser.minimize(loss)

#Timestamps for logging and filenames.
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "logs"
logdir = "{}/run-{}".format(root_logdir, now)

#Used for TensorBoard, the way to view the graph of the network.
mse_summary = tf.summary.scalar('MSE', loss)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())


#Initialise the nodes in the network. 
init = tf.global_variables_initializer()

#Hyperparameters and paths for the outputs of the network.
n_epochs = 1000000000
batch_size = 20
saver = tf.train.Saver()
best_path = "model/best_model"
current_path = "model/current_path"

#Takes the output of the network and converts it to midi.
def Output(epoch, current_path, sess):
    saver.restore(sess, current_path)
    seed = get_batches(1)
    X_batch = np.array(seed).reshape(1, n_steps, n_outputs)
    y_pred = sess.run(outputs, feed_dict={X: X_batch})
    for line in y_pred[0]:
        if(os.path.isfile("outputs/" + str(epoch) + "_" + now + ".csv")):
            with open("outputs/" + str(epoch) + "_" + now + ".csv", "a") as output_file:
                output_file.write(str(round(line[0])) + "," + str(round(line[1])) + "\n")
        else:    
            with open("outputs/" + str(epoch) + "_" + now +".csv", "w+") as output_file:
                output_file.write(str(round(line[0])) + "," + str(round(line[1])) + "\n")
    #Convert to midi
    csv2midi.Work("outputs/" + str(epoch) + "_" + now + ".csv")


#Runs the network, the training phase.
with tf.Session() as sess:
    #Quick way to use pretrained network. Very useful for making small changes and not having to wait a few thousand epochs before having coherent music generated.
    response = input("Do you want to load the last used model? Y/N").upper()
    if(response == "Y"):
        saver.restore(sess, current_path)         
    else:
        init.run()
    best_mse = np.infty
    for epoch in range(n_epochs):
        X_batch, y_batch = get_batches(batch_size)
        sess.run(training_op, feed_dict={X: X_batch, y: y_batch}) 
        if epoch % 100 == 0:
            saver.save(sess, current_path)
            Output(epoch, current_path, sess)
            summary_str = mse_summary.eval(feed_dict={X: X_batch, y: y_batch})
            mse = loss.eval(feed_dict={X: X_batch, y: y_batch})
            print(epoch, "MSE:", mse)
            if mse < best_mse:
                best_mse = mse
                print("New best model")
                sample = outputs.eval(feed_dict={X: X_batch, y: y_batch})
                print("sample:", sample[0] )
                saver.save(sess, best_path)
