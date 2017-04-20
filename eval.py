
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from source_cnn import SourceCNN
from tensorflow.contrib import learn
import csv
import re
# Parameters
# ==================================================

# Data Parameters
tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the positive data.")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")
tf.flags.DEFINE_string("test_code", "", "Checkpoint directory from training run")


# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

tokenSplit = r'[\w\']+|[""!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~""\\]'
def tokenizer(docs):
    return [re.findall(tokenSplit, doc) for doc in docs]


# CHANGE THIS: Load data. Load your own data here
if FLAGS.eval_train:
    x_raw, y_test = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
    y_test = np.argmax(y_test, axis=1)
else:
    x_raw = [FLAGS.test_code]
    y_test = None
# Map data into vocabulary
vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_processor.transform(x_raw)))

print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]
        scores = graph.get_operation_by_name("output/scores").outputs[0]
        # Generate batches for one epoch
        batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)
        # print(predictions)
        # Collect the predictions here
        all_predictions = []
        # print(np.array(list(batches)))
        for x_test_batch in batches:
            label = ["javascript", "java", "cpp","csharp","python","ruby", "objc"]
            batch_predictions = sess.run([predictions,scores], {input_x: x_test_batch, dropout_keep_prob: 1.0})
            # print(batch_predictions[1])
            print('Prediction: ',label[batch_predictions[0][0]], 'score :',np.exp(max(batch_predictions[1][0]))/np.sum(np.exp(batch_predictions[1][0])))
            all_predictions = np.concatenate([all_predictions, batch_predictions[0]])

# Print accuracy if y_test is defined
if y_test is not None:
    # print(all_predictions)
    correct_predictions = float(sum(all_predictions == y_test))
    print(label[correct_predictions])
    print("Total number of test examples: {}".format(len(y_test)))
    print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))

# Save the evaluation to a csv
predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
print("Saving evaluation to {0}".format(out_path))
with open(out_path, 'w') as f:
    csv.writer(f).writerows(predictions_human_readable)
