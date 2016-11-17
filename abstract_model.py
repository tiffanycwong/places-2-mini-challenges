import tensorflow as tf
import numpy as np
import math
import tensorflow.contrib.slim as slim
import time
import utils
import os

class AbstractModel():

    def __init__(self, FLAGS, session):
        self.FLAGS = FLAGS
        self.session = session
        self.build_model_and_architecture(FLAGS)

    def build_model(self):
        raise NotImplementedError("Please Implement function")

    def get_optimizer(self):
        raise NotImplementedError("Please Implement function")

    def get_losses(self):
        raise NotImplementedError("Please Implement function")

    def get_eval_metrics(self):
        raise NotImplementedError("Please Implement function")

    def get_eval_metric_names(self):
        raise NotImplementedError("Please Implement function")

    @staticmethod
    def model_name():
        raise NotImplementedError("Please Implement function")        

    def build_model_and_architecture(self, FLAGS):
        print("Building model...")

        self.input_placeholder = tf.placeholder(tf.float32, [None, 299, 299, 3], name='input_placeholder')

        self.label_placeholders_dict = {}
        self.label_placeholders_dict["scene_category"] = tf.placeholder(tf.float32, [None, 100], name='scene_category')
        # self.label_placeholders_dict["objects"] = tf.placeholder(tf.float32, [None, 100], name='obj')

        self.keep_prob = tf.placeholder(tf.float32)

        self.outputs = self.build_model()

        self.losses = self.get_losses()
        self.eval_metrics = self.get_eval_metrics()
        self.loss = self.get_total_loss(self.losses)
        self.optimizer = self.get_optimizer()

        self.saver = tf.train.Saver()

    def image_dimensions(self):
        return (3, 128, 128)

    def get_total_loss(self, losses):
        total_loss = sum(losses)
        # total_loss = slim.losses.get_total_loss(total_loss, add_regularization_losses=True)
        return total_loss

    def train(self, FLAGS, X_train, Y_train, X_val, Y_val):
        tf.initialize_all_variables().run()
        counter = 0
        for epoch in xrange(FLAGS.epoch):
            start_time = time.time()
            self.get_all_metrics_post_epoch(X_train, Y_train, X_val, Y_val, counter, epoch, start_time)

            X_train, Y_train = utils.shuffle_order(X_train, Y_train)
            n_batches = int(math.ceil(1.0*X_train.shape[0] / self.FLAGS.batch_size))

            for batch_i in xrange(0, n_batches):
                batch_input, batch_labels = utils.get_batch(X_train, Y_train, batch_size=FLAGS.batch_size, batch_index=batch_i, augment_training_data=FLAGS.augment_training_data)

                feed_dict = {self.input_placeholder: batch_input, self.keep_prob:FLAGS.keep_prob}

                feed_dict[self.label_placeholders_dict["scene_category"]] = batch_labels["scene_category"]

                for key, labels in batch_labels.iteritems():
                    placeholder = self.label_placeholders_dict[key]
                    feed_dict[placeholder] = np.array(labels)

                # Update Network Parameters and get Summary
                _, batch_loss = self.session.run([self.optimizer, self.loss],
                    feed_dict=feed_dict)
                if self.FLAGS.verbose:
                    print("Batch: [{:2d}] Finished, Stats: time: {:4.4f}, loss: {:.8f}".format(batch_i, time.time() - start_time, batch_loss))

                counter += 1

                if np.mod(counter, 500) == 2:
                    print("Saving Model")
                    if False:
                        self.save(FLAGS.checkpoint_dir, counter)

    def save(self, checkpoint_dir, step):
        model_name = "{}.model".format(self.model_name())

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.session,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.session, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def get_all_metrics_post_epoch(self, X_train, Y_train, X_val, Y_val, counter, epoch, start_time):
        self.get_metrics_post_epoch('train', X_train, Y_train, self.FLAGS.batch_size, counter, epoch, start_time)
        self.get_metrics_post_epoch('val', X_val, Y_val, self.FLAGS.batch_size, counter, epoch, start_time)

    def get_metrics_post_epoch(self, group, X, Y, batch_size, counter, epoch, start_time):
        batch_acc_array = []
        batch_loss_array = []
        batch_lengths = []
        start_time = time.time()

        n_batches = int(math.ceil(1.0*X.shape[0] / self.FLAGS.batch_size))

        eval_metric_names = self.get_eval_metric_names()
        metrics_tally = [[] for key in eval_metric_names]
        total_examples = 0.0

        for batch_i in xrange(0, n_batches):
            batch_input, batch_labels = utils.get_batch(X, Y, batch_size=self.FLAGS.batch_size, batch_index=batch_i, augment_training_data=self.FLAGS.augment_training_data)

            feed_dict = {self.input_placeholder: batch_input, self.keep_prob:1.0}

            feed_dict[self.label_placeholders_dict["scene_category"]] = batch_labels["scene_category"]

            for key, labels in batch_labels.iteritems():
                placeholder = self.label_placeholders_dict[key]
                feed_dict[placeholder] = np.array(labels)

            # Update Network Parameters and get Summary
            eval_metrics = self.session.run(self.eval_metrics, feed_dict=feed_dict)

            for i in range(len(eval_metrics)):
                metric_value = eval_metrics[i]
                metric_name = eval_metric_names[i]
                metrics_tally[i].append(metric_value*len(batch_input))
            total_examples += len(batch_input)

        for i in range(len(metrics_tally)):
            metric_value = np.sum(metrics_tally[i])/total_examples
            metric_name = eval_metric_names[i]
            print("Epoch [{}] [{}] {} = {:.8f}".format(epoch, group, metric_name, metric_value))
        print("")


    def evaluate_model(self, FLAGS, X_train, Y_train, X_val, Y_val):
        tf.initialize_all_variables().run()
        counter = 1
        epoch = 0
        start_time = time.time()
        self.get_metrics_post_epoch('val', X_val, Y_val, self.FLAGS.batch_size, counter, epoch, start_time)
