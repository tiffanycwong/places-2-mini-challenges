import numpy as np
import pprint
import tensorflow as tf
import os

import utils

# EDIT THIS to change which Model gets loaded
from shallow_model_with_word_embeddings import Model as MODEL
#from shallow_model import Model as MODEL
# from shallow_model_with_word_embeddings import Model as MODEL
# from inception_model import Model as MODEL

flags = tf.app.flags
flags.DEFINE_integer("epoch", 200, "Epochs to train")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate of for adam")
flags.DEFINE_float("keep_prob", 0.75, "Dropout probability to keep")
flags.DEFINE_integer("batch_size", 128, "The size of mini batches to use")
flags.DEFINE_string("checkpoint_dir", "checkpoints/{}".format(MODEL.model_name()), "Directory name to save the checkpoints")
flags.DEFINE_boolean("load_small", True, "True for small data (makes debugging easier).")
flags.DEFINE_boolean("load_easy", False, "True for easy data (just 4 classes) (makes debugging easier).")
flags.DEFINE_boolean("is_train", True, "True for training, False for testing")
flags.DEFINE_boolean("augment_training_data", False, "True to augment training data, False to leave alone")
flags.DEFINE_boolean("verbose", False, "True to print verbose-style")
flags.DEFINE_boolean("restart_training", True, "True to restart training from scratch everytime, False to load from last checkpoint.")
flags.DEFINE_integer("resize_dim", 128, "Set dimension to resize image to.")

FLAGS = flags.FLAGS

def main(_):
    with tf.Session() as session:
        print('Starting Session...')
        model = MODEL(FLAGS, session)
        if FLAGS.is_train:
            model.train()
        else:
            # model.load(FLAGS.checkpoint_dir)
            model.evaluate_on_test()

if __name__ == '__main__':
    tf.app.run()
