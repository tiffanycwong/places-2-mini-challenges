import tensorflow as tf
import numpy as np
from abstract_model import AbstractModel
import tensorflow.contrib.slim as slim
# from tensorflow.contrib.slim.nets import inception_v3
# from tensorflow.contrib.slim.python.slim.nets import inception_v3
from slim_models.nets import inception_v3

class Model(AbstractModel):
    def build_model(self):
        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            # Get Inception Architecture
            print(self.input_placeholder.get_shape().as_list())
            logits, end_points = inception_v3.inception_v3(self.input_placeholder, num_classes=1001)

            # # Load Pretrained Inception Weights
            # loader = tf.train.Saver()
            # loader.restore(self.session, "./pretrained_slim/inception_v3.ckpt")

            # Get image features output from Inception and make new classifier layers
            image_features = slim.flatten(end_points['PreLogits'])

            scene_logits = slim.fully_connected(image_features, 100, activation_fn=None, scope='birads_logits')

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
        
        # return [scene_loss, scene_accuracy]
        return [scene_loss, scene_accuracy, in_top_5]

    def get_eval_metric_names(self):
        # return ["Scene Loss", "Scene Accuracy Top 1"]
        return ["Scene Loss", "Scene Accuracy Top 1", "Scene Accuracy Top 5"]

    def get_output_names(self):
        return ["Scene Classification"]

    def get_optimizer(self):
        optimizer = tf.train.AdamOptimizer(self.FLAGS.learning_rate).minimize(self.loss)
        return optimizer

    @staticmethod
    def model_name():
        return "AlexNetModel"
