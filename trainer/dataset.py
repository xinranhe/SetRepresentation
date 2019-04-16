import random
import tensorflow as tf

def get_parse_spec():
    parse_spec = {
        "movie_seq": tf.VarLenFeature(tf.int64),
        "rate_seq": tf.VarLenFeature(tf.int64),
        "label": tf.FixedLenFeature([1], tf.int64),
    }
    return parse_spec


def _tfrecord_parse_fn(example_proto):
    parsed_features = tf.parse_single_example(example_proto, get_parse_spec())
    return parsed_features, parsed_features["label"]


def input_function(filename_patterns, is_train, parameters):
    parse_spec = get_parse_spec()

    def input_fn():
        input_files = []
        for filename_pattern in filename_patterns:
            input_files.extend(tf.gfile.Glob(filename_pattern))
        if is_train:
            random.shuffle(input_files)
        files = tf.data.Dataset.list_files(input_files)
        dataset = files.apply(tf.contrib.data.parallel_interleave(
            lambda fs: tf.data.TFRecordDataset(fs, compression_type=parameters["compression_type"]), cycle_length=parameters["num_parallel_readers"]))
        if is_train:
            dataset = dataset.shuffle(parameters["buffer_size"])
        dataset = dataset.map(_tfrecord_parse_fn, num_parallel_calls=parameters["num_parsing_threads"])
        dataset = dataset.batch(parameters["batch_size"])
        dataset = dataset.prefetch(buffer_size=parameters["prefetch_buffer_size"])
        dataset = dataset.repeat()
        return dataset
    return input_fn
