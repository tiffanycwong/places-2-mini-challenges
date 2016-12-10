import tensorflow as tf
import numpy as np
from abstract_model import AbstractModel
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.python.slim.nets.vgg as vgg

class Model(AbstractModel):
    def build_model(self):
        is_train = self.FLAGS.is_train
        dropout_keep_prob = 1.0
        if is_train:
            dropout_keep_prob = 0.5

        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                          activation_fn=tf.nn.relu,
                          weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                          weights_regularizer=slim.l2_regularizer(0.0005)):
            net = slim.repeat(self.input_placeholder, 2, slim.conv2d, 64, [3, 3], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
            net = slim.max_pool2d(net, [2, 2], scope='pool5')
            net = slim.flatten(net, scope="flatten")
            net = slim.fully_connected(net, 2048, scope='fc6')
            net = slim.dropout(net, dropout_keep_prob, scope='dropout6')
            net = slim.fully_connected(net, 2048, scope='fc7')
            net = slim.dropout(net, dropout_keep_prob, scope='dropout7')
            image_features = net

            scene_logits = slim.fully_connected(image_features, 100, activation_fn=None, scope='scene_pred', trainable=True)
            multi_hot_logits = slim.fully_connected(image_features, 175, activation_fn=None, scope='multi_hot_logits', trainable=True)
            word_embedding_logits = slim.fully_connected(image_features, 300, activation_fn=None, scope='word_embedding_pred', trainable=True)

            obj_embedding_size = 40
            object_embedding_logits = slim.fully_connected(image_features, obj_embedding_size, activation_fn=None, scope='object_embedding_pred', trainable=True)

            outputs = [scene_logits, multi_hot_logits, word_embedding_logits, object_embedding_logits]

        return outputs

    def get_losses(self):
        scene_logits = self.outputs[0]
        scene_labels_placeholder = self.label_placeholders_dict['scene_category']
        scene_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(scene_logits, scene_labels_placeholder))

        object_multihot_logits = self.outputs[1]
        object_multihot_labels_placeholder = self.label_placeholders_dict['object_multihot']
        object_multihot_embedding_loss = tf.reduce_mean(tf.sqrt(tf.square(object_multihot_labels_placeholder-object_multihot_logits)))

        word_embedding_logits = self.outputs[2]
        word_embedding_labels_placeholder = self.label_placeholders_dict['word_embeddings_averages']
        word_embedding_loss = tf.reduce_mean(tf.sqrt(tf.square(word_embedding_logits-word_embedding_labels_placeholder)))

        object_embedding_logits = self.outputs[3]
        object_embedding_labels_placeholder = self.label_placeholders_dict['compressed_object_encodings']
        object_embedding_loss = tf.reduce_mean(tf.sqrt(tf.square(object_embedding_logits-object_embedding_labels_placeholder)))

        losses = [scene_loss, object_embedding_loss, object_multihot_embedding_loss, word_embedding_loss]

        return losses

    def get_eval_metrics(self):
        scene_logits = self.outputs[0]
        scene_labels_placeholder = self.label_placeholders_dict['scene_category']

        scene_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(scene_logits, scene_labels_placeholder))

        correct_scene_prediction = tf.equal(tf.argmax(scene_logits, 1), tf.argmax(scene_labels_placeholder,1))
        scene_accuracy = tf.reduce_mean(tf.cast(correct_scene_prediction, "float"))

        in_top_5 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(scene_logits, tf.argmax(scene_labels_placeholder, 1), 5), "float"))

        object_multihot_logits = self.outputs[1]
        object_multihot_labels_placeholder = self.label_placeholders_dict['object_multihot']
        object_multihot_embedding_loss = tf.reduce_mean(tf.sqrt(tf.square(object_multihot_logits-object_multihot_labels_placeholder)))

        word_embedding_logits = self.outputs[2]
        word_embedding_labels_placeholder = self.label_placeholders_dict['word_embeddings_averages']
        word_embedding_loss = tf.reduce_mean(tf.sqrt(tf.square(word_embedding_logits-word_embedding_labels_placeholder)))

        object_embedding_logits = self.outputs[3]
        object_embedding_labels_placeholder = self.label_placeholders_dict['compressed_object_encodings']
        object_embedding_loss = tf.reduce_mean(tf.sqrt(tf.square(object_embedding_logits-object_embedding_labels_placeholder)))
        
        return [scene_loss, scene_accuracy, in_top_5, object_multihot_embedding_loss, word_embedding_loss, object_embedding_loss]

    def get_eval_metric_names(self):
        return ["Scene Loss", "Scene Accuracy Top 1", "Scene Accuracy Top 5", "Multi hot loss", "Word embedding loss", "Compressed Object Encodings loss"]

    def get_output_names(self):
        return ["Scene Classification", "Multihot Prediction", "Word Embedding Prediction", "Compressed Object Encodings"]

    def get_optimizer(self):
        optimizer = tf.train.GradientDescentOptimizer(self.FLAGS.learning_rate).minimize(self.loss)
        # optimizer = tf.train.AdamOptimizer(self.FLAGS.learning_rate).minimize(self.loss)
        return optimizer

    def get_loss_weights(self):
        return self.FLAGS.loss_weights

    @staticmethod
    def model_name():
        return "VGGModelFast"
