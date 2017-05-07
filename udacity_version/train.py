#!/usr/bin/env python
from __future__ import division
import time
import tensorflow as tf


class Trainer(object):

    def __init__(self, agent):
        self.agent = agent
        self.env = agent.env
        self.saver = tf.train.Saver()

    def run(self):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            self.agent.randomRestart()

            successes = 0
            failures = 0
            total_loss = 0

            print "starting %d random plays to populate replay memory" % self.agent.replay_start_size
            for i in xrange(self.agent.replay_start_size):
                # follow random policy
                state, action, reward, next_state, terminal = self.agent.observe(1)

                if reward == 1:
                    successes += 1
                elif terminal:
                    failures += 1

                if (i+1) % 10000 == 0:
                    print "\nmemory size: %d" % len(self.agent.memory),\
                          "\nSuccesses: ", successes,\
                          "\nFailures: ", failures
            
            sample_success = 0
            sample_failure = 0
            print "\nstart training..."
            start_time = time.time()
            loop_start_time = time.time()
            for i in xrange(self.agent.train_steps):
                # annealing learning rate
                current_step = i+1
                lr = self.agent.trainEps(i)
                state, action, reward, next_state, terminal = self.agent.observe(lr)

                if len(self.agent.memory) > self.agent.batch_size and current_step % self.agent.update_freq == 0:
                    sample_success, sample_failure, loss = self.agent.doMinibatch(sess, sample_success, sample_failure)
                    total_loss += loss

                if current_step % self.agent.steps == 0:
                    self.agent.copy_weights(sess)

                if reward == 1:
                    successes += 1
                elif terminal:
                    failures += 1
                
                if (current_step % self.agent.save_weights == 0):
                    self.agent.save(self.saver, sess, current_step)

                if (current_step % self.agent.batch_size == 0):
                    avg_loss = total_loss / self.agent.batch_size
                    end_time = time.time()
                    elapsed_time = end_time-start_time
                    print "\nTraining step: ", current_step, " of ", self.agent.train_steps,\
                          "\nMemory size: ", len(self.agent.memory),\
                          "\nLearning rate: ", lr,\
                          "\nSuccesses: ", successes,\
                          "\nFailures: ", failures,\
                          "\nSample successes: ", sample_success,\
                          "\nSample failures: ", sample_failure,\
                          "\nAverage batch loss: ", avg_loss,\
                          "\nElapsed time: ", elapsed_time, "s"\
                          "\nBatch training time: ", (end_time-loop_start_time)/self.agent.batch_size, "s"\
                          "\nRemaining time: ", ((elapsed_time/current_step)*self.agent.train_steps), "s"
                    loop_start_time = time.time()
                    total_loss = 0
