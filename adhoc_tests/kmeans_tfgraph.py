import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.factorization import KMeans

# Read Data
mnist = input_data.read_data_sets("/data/", one_hot=True)

# hparams
batch_size = 1024
num_steps = 25
k = 25  # num of clusters
num_classes = 10

X = tf.placeholder(dtype=tf.float32, shape=[None, 784])
Y = tf.placeholder(dtype=tf.float32, shape=[None, num_classes])

# Kmeans Init
kmeans = KMeans(inputs=X, num_clusters=k, use_mini_batch=True)
all_scores, cluster_idx, scores, cluster_centers_initialized, \
    cluster_centers_var, init_op, training_op = kmeans.training_graph()
cluster_index = cluster_idx[0]  # unwrap


avg_distance = tf.reduce_mean(all_scores)  # This is the minimization target
init_param = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_param)
    sess.run(init_op, feed_dict={X: mnist.train.images})

    for iter in range(1, num_steps+1):
        _, d, idx = sess.run([training_op, avg_distance, cluster_index], feed_dict={X: mnist.train.images})
        if iter % 10 ==0 or iter ==1:
            print("Step: %d, distance=%f" %(iter, d))

    counts = np.zeros (shape=(k, num_classes))
    for i in range (len (idx)):
        counts[idx[i]] += mnist.train.labels[i]
    labels_map = [np.argmax (c) for c in counts]
    labels_map = tf.convert_to_tensor (labels_map)


    cluster_label = tf.nn.embedding_lookup (labels_map, cluster_index)

    # Compute accuracy
    correct_prediction = tf.equal(cluster_label, tf.cast (tf.argmax (Y, 1), tf.int32))
    accuracy_op = tf.reduce_mean (tf.cast (correct_prediction, tf.float32))

    # Test Model
    test_x, test_y = mnist.test.images, mnist.test.labels
    print ("Test Accuracy:", sess.run (accuracy_op, feed_dict={X: test_x, Y: test_y}))