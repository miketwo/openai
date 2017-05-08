#!/usr/bin/env python
'''
From:
https://www.tensorflow.org/get_started/get_started
'''
import tensorflow as tf
import time


def main():
    # Model parameters
    W = tf.Variable([.3], tf.float32)
    b = tf.Variable([-.3], tf.float32)
    # Model input and output
    x = tf.placeholder(tf.float32)
    linear_model = W * x + b
    y = tf.placeholder(tf.float32)
    # loss
    loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares
    # optimizer
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)
    # training data
    x_train = [1,2,3,4]
    y_train = [0,-1,-2,-3]
    # training loop
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init) # reset values to wrong
    start = time.time()
    for i in range(1000):
      sess.run(train, {x:x_train, y:y_train})
    end = time.time()

    # evaluate training accuracy
    curr_W, curr_b, curr_loss  = sess.run([W, b, loss], {x:x_train, y:y_train})
    print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
    print("Training time: {}s".format(end-start))


if __name__ == '__main__':
    main()