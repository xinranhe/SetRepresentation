import importlib
import tensorflow as tf

from trainer import dataset
from trainer import model

FLAGS = tf.app.flags.FLAGS
tf.flags.DEFINE_string("model_config_path", "test", "File name for model parameter configuration")

tf.logging.set_verbosity(tf.logging.INFO)

def main(unused_argv):
    params = importlib.import_module('trainer.params.' + FLAGS.model_config_path)
    PARAMETERS = params.PARAMETERS

    train_input_fn = dataset.input_function(PARAMETERS["training_data_patterns"], "train", PARAMETERS)
    eval_input_fn = dataset.input_function(PARAMETERS["evaluation_data_patterns"], "eval", PARAMETERS)

    if PARAMETERS["num_gpus"] <= 1:
        estimator = tf.estimator.Estimator(model_fn=model.model_fn, model_dir=PARAMETERS["model_dir"], params=PARAMETERS)
    else:
        devices = ["device:GPU:%d" % i for i in range(PARAMETERS["num_gpus"])]
        distribution_strategy = tf.contrib.distribute.MirroredStrategy(devices=devices)
        estimator = tf.estimator.Estimator(model_fn=model.model_fn, model_dir=PARAMETERS["model_dir"], params=PARAMETERS, 
            config=tf.estimator.RunConfig(train_distribute=distribution_strategy))

    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=PARAMETERS["num_train_steps"])

    exporter = tf.estimator.LatestExporter('exporter', model.serving_input_receiver_fn(PARAMETERS), exports_to_keep=None)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, steps=PARAMETERS["num_eval_steps"], exporters=exporter)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == '__main__':
    tf.app.run()
