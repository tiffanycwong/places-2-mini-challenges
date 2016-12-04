import tensorflow as tf
import numpy as np
import math
import tensorflow.contrib.slim as slim
import time
import utils
import os
from skimage import io

class AbstractModel():

    def __init__(self, FLAGS, session):
        self.FLAGS = FLAGS
        self.session = session
        self.build_model_and_architecture()

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

    def get_loss_weights(self):
        return [1.0]

    def build_model_and_architecture(self):
        print("Building model...")

        self.input_placeholder = tf.placeholder(tf.float32, [None, self.FLAGS.resize_dim, self.FLAGS.resize_dim, 3], name='input_placeholder')

        self.label_placeholders_dict = {}
        self.label_placeholders_dict["scene_category"] = tf.placeholder(tf.float32, [None, 100], name='scene_category')
        n_objects = 1000 # replace with different number if not true
        word_embedding_size = 300
        self.label_placeholders_dict["object_encodings"] = tf.placeholder(tf.float32, [None, n_objects], name='object_encodings')
        self.label_placeholders_dict["word_embeddings_averages"] = tf.placeholder(tf.float32, [None, word_embedding_size], name='word_embeddings_averages')

        self.keep_prob = tf.placeholder(tf.float32)

        self.outputs = self.build_model()

        self.losses = self.get_losses()
        self.eval_metrics = self.get_eval_metrics()
        self.loss = self.get_total_loss(self.losses)
        self.optimizer = self.get_optimizer()

        self.num_train_examples = len(utils.get_image_path_label_pairs('train', self.FLAGS.load_easy, self.FLAGS.load_small))
        self.num_val_examples = len(utils.get_image_path_label_pairs('val', self.FLAGS.load_easy, self.FLAGS.load_small))

        self.num_test_examples = len(utils.get_test_image_paths())
        self.train_indices = range(self.num_train_examples)
        self.saver = tf.train.Saver()

    def get_total_loss(self, losses):
        weights = self.get_loss_weights()
        for i in range(len(losses)):
            loss = losses[i]
            weight = weights[i]
            if i == 0:
                total_loss = loss * weight
            else:
                total_loss += loss * weight
        
        return total_loss

    def train(self):
        tf.initialize_all_variables().run()
        counter = 0
        for epoch in xrange(self.FLAGS.epoch):
            self.train_indices = utils.shuffle_order(self.train_indices)
            start_time = time.time()
            self.get_all_metrics_post_epoch(counter, epoch, start_time)

            n_batches = int(math.ceil(1.0*self.num_train_examples/self.FLAGS.batch_size))

            for batch_i in xrange(0, n_batches):
                batch_input, batch_labels = utils.get_data_for_batch('train', batch_i, self.FLAGS.batch_size, self.train_indices, self.FLAGS.load_small, self.FLAGS.load_easy, self.FLAGS.resize_dim)

                feed_dict = {self.input_placeholder: batch_input, self.keep_prob:self.FLAGS.keep_prob}

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
                        self.save(self.FLAGS.checkpoint_dir, counter)

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

    def get_all_metrics_post_epoch(self, counter, epoch, start_time):
        self.get_metrics_post_epoch('train', self.FLAGS.batch_size, counter, epoch, start_time)
        self.get_metrics_post_epoch('val', self.FLAGS.batch_size, counter, epoch, start_time)

    def get_metrics_post_epoch(self, split_name, batch_size, counter, epoch, start_time):
        batch_acc_array = []
        batch_loss_array = []
        batch_lengths = []
        start_time = time.time()

        if split_name is 'train':
            num_examples = self.num_train_examples
        elif split_name is 'val':
            num_examples = self.num_val_examples

        n_batches = int(math.ceil(1.0*num_examples / self.FLAGS.batch_size))

        for batch_i in xrange(0, n_batches):
            # batch_input, batch_labels = utils.get_batch(X, Y, batch_size=self.FLAGS.batch_size, batch_index=batch_i, augment_training_data=self.FLAGS.augment_training_data)
            batch_input, batch_labels = utils.get_data_for_batch(split_name, batch_i, self.FLAGS.batch_size, range(num_examples), self.FLAGS.load_small, self.FLAGS.load_easy, self.FLAGS.resize_dim)

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
            print("Epoch [{}] [{}] {} = {:.8f}".format(epoch, split_name, metric_name, metric_value))
        print("")


    def evaluate_on_test(self):
        tf.initialize_all_variables().run()
        self.predict_and_output_to_file()


    def predict_and_output_to_file(self):
        image_file_names = utils.get_test_image_paths()
        image_file_names = image_file_names[:200]
        predictions = []

        with open ('./output_{}.txt'.format(self.model_name()), 'w') as file:

            for example_index in range(len(image_file_names)):
                if example_index % 100 == 0:
                    print("{} of {}".format(example_index, len(image_file_names)))
                file_name = image_file_names[example_index]
                image = io.imread(file_name)
                image = utils.image_process(image, 128)

                feed_dict = {self.input_placeholder: np.array([image]), self.keep_prob:1.0}

                # Update Network Parameters and get Summary
                prediction = self.session.run(self.outputs[0], feed_dict=feed_dict)
                prediction = prediction[0]
                prediction_list = [(prediction[index], index) for index in range(len(prediction))]
                sorted_prediction_list = sorted(prediction_list)[::-1]
                best_5_indicies = sorted_prediction_list[0:5]
                best_5_indicies = [best[1] for best in best_5_indicies]
                predictions.append(best_5_indicies)
                best_5_string = [str(index) for index in best_5_indicies]
                best_5_string = " ".join(best_5_string)
                file_name_plain = file_name.split('/')[-1]

                file.write(file_name_plain + " " + best_5_string + "\n")

        file.close()

