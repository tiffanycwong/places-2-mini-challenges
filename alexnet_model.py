import tensorflow as tf
import numpy as np
from abstract_model import AbstractModel
import tensorflow.contrib.slim as slim

class Model(AbstractModel):
    def build_model(self):
        net = slim.conv2d(self.input_placeholder, 64, [11, 11], 4, padding='VALID',
                        scope='conv1')
        net = slim.max_pool2d(net, [3, 3], 2, scope='pool1')
        net = slim.conv2d(net, 192, [5, 5], scope='conv2')
        net = slim.max_pool2d(net, [3, 3], 2, scope='pool2')
        net = slim.conv2d(net, 384, [3, 3], scope='conv3')
        net = slim.conv2d(net, 384, [3, 3], scope='conv4')
        net = slim.conv2d(net, 256, [3, 3], scope='conv5')
        net = slim.max_pool2d(net, [3, 3], 2, scope='pool5')

        with slim.arg_scope([slim.conv2d],
                          weights_initializer=tf.truncated_normal_initializer(0.0, 0.005),
                          biases_initializer=tf.constant_initializer(0.1)):
            net = slim.conv2d(net, 4096, [2, 2], padding='VALID',
                              scope='fc6')
            net = slim.dropout(net, self.keep_prob,
                               scope='dropout6')
            net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
            net = slim.dropout(net, self.keep_prob,
                               scope='dropout7')
            scene_logits = tf.squeeze(slim.conv2d(net, 100, [1, 1],
                              activation_fn=None,
                              normalizer_fn=None,
                              biases_initializer=tf.zeros_initializer,
                              scope='fc8'))

        outputs = [scene_logits]
        return outputs

    def get_losses(self):
        scene_logits = self.outputs[0]
        scene_labels_placeholder = self.label_placeholders_dict['scene_category']
        scene_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(scene_logits, scene_labels_placeholder))
        losses = [scene_loss]
        return losses

    def get_eval_metrics(self):
        scene_logits = self.outputs[0]
        scene_labels_placeholder = self.label_placeholders_dict['scene_category']

        scene_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(scene_logits, scene_labels_placeholder))

        correct_scene_prediction = tf.equal(tf.argmax(scene_logits, 1), tf.argmax(scene_labels_placeholder,1))
        scene_accuracy = tf.reduce_mean(tf.cast(correct_scene_prediction, "float"))

        in_top_5 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(scene_logits, tf.argmax(scene_labels_placeholder, 1), 5), "float"))
        
        return [scene_loss, scene_accuracy, in_top_5]

    def get_eval_metric_names(self):
        return ["Scene Loss", "Scene Accuracy Top 1", "Scene Accuracy Top 5"]

    def get_output_names(self):
        return ["Scene Classification"]

    def get_optimizer(self):
        optimizer = tf.train.AdamOptimizer(self.FLAGS.learning_rate).minimize(self.loss)
        return optimizer

    @staticmethod
    def model_name():
        return "AlexNetModel"
