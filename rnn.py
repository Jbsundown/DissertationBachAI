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
batch_sequence_lengths_train = np.asarray([len(x) for x in X_train])
n_steps = 40
n_inputs = 2
n_neurons = 400
n_outputs = 2
n_output_steps = 40
n_layers = 6
eta = 0.0003

def get_batches(batch_size):
    #batch_size is the amount of times we go through n_steps. So a bs of 10 would make us go through a list of 10 where X is a list of 50x2
    #randomly generate a list of numbers to choose from for building the batch.
    starting = np.random.permutation(len(X_train))
    batch_test = starting[:batch_size]
    batch_list = np.asarray([X_train[x] for x in batch_test])
    if(batch_size == 1):
        return batch_list[:, :40].reshape(-1, n_steps, n_inputs)
    return batch_list[:, :40].reshape(-1, n_steps, n_inputs), batch_list[:, 1:].reshape(-1, n_steps, n_outputs)

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder (tf.float32, [None, n_output_steps, n_outputs])
#sequence_length = tf.placeholder(tf.int32, [None])
gru_cell = [tf.nn.rnn_cell.GRUCell(n_neurons) for layer in range (n_layers)]
multi_cell = tf.nn.rnn_cell.MultiRNNCell(gru_cell)
rnn_outputs, states = tf.nn.dynamic_rnn(multi_cell, X, dtype=tf.float32)#, sequence_length=sequence_length

stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, n_neurons])
stacked_outputs = tf.layers.dense(stacked_rnn_outputs, n_outputs)
outputs = tf.reshape(stacked_outputs, [-1, n_output_steps, n_outputs])
loss = tf.reduce_mean(tf.square(outputs - y))
optimiser = tf.train.AdamOptimizer(learning_rate=eta)
training_op = optimiser.minimize(loss)

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "logs"
logdir = "{}/run-{}".format(root_logdir, now)

mse_summary = tf.summary.scalar('MSE', loss)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
init = tf.global_variables_initializer()

n_epochs = 10000000
batch_size = 10
saver = tf.train.Saver()
best_path = "model/best_model"
current_path = "model/current_path"


def Output(epoch, current_path):
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


with tf.Session() as sess:
    init.run()
    best_mse = np.infty
    for epoch in range(n_epochs):
        X_batch, y_batch = get_batches(batch_size)#, seq_length
        sess.run(training_op, feed_dict={X: X_batch, y: y_batch}) #sequence_length: seq_length
        if epoch % 100 == 0:
            saver.save(sess, current_path)
            Output(epoch, current_path)
            summary_str = mse_summary.eval(feed_dict={X: X_batch, y: y_batch})#, sequence_length: seq_length
            file_writer.add_summary(summary_str, epoch)
            mse = loss.eval(feed_dict={X: X_batch, y: y_batch})#, sequence_length: seq_length
            print(epoch, "MSE:", mse)
            if mse < best_mse:
                best_mse = mse
                print("New best model")
                sample = outputs.eval(feed_dict={X: X_batch, y: y_batch})#, sequence_length: seq_length
                #ys = y.eval(feed_dict={X: X_batch, y: y_batch})#, sequence_length: seq_length
                print("sample:", sample[0] )
                #print("y:", ys[0])
                saver.save(sess, best_path)
FileWriter.close()