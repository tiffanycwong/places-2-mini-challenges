import tensorflow as tf
import numpy as np
from abstract_model import AbstractModel
import tensorflow.contrib.slim as slim

class Model(AbstractModel):
    def build_model(self):
        net = slim.conv2d(self.input_placeholder, 64, [5, 5], stride=2, scope='conv1', activation_fn=tf.nn.relu, trainable=True)
        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        net = slim.conv2d(self.input_placeholder, 256, [5, 5], stride=2, scope='conv2', activation_fn=tf.nn.relu, trainable=True)
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        net = slim.flatten(net, scope='flatten')
        net = slim.fully_connected(net, 512, activation_fn=tf.nn.relu, scope='fc1', trainable=True)

        scene_logits = slim.fully_connected(net, 100, activation_fn=None, scope='birads_pred', trainable=True)
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
        optimizer = tf.train.GradientDescentOptimizer(self.FLAGS.learning_rate).minimize(self.loss)
        return optimizer

    @staticmethod
    def model_name():
        return "ShallowModel"
