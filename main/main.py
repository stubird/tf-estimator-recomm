"""
   Copyright 2018 (c) Jinxin Xie

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from estimator_callback.TEP import *
from data_loader.data_input import *
import argparse

filepath="F:\\studycode\\tensorflow\\pyrecommd\\TF-recomm\\ml-1m\\ratings.dat"

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=1000, type=int,
                    help='batch size')
parser.add_argument('--train_steps', default=10000, type=int,
                    help='number of training steps')
parser.add_argument('--datafile', default=filepath, type=str,
                    help='path of data')
parser.add_argument('--dim', default=15, type=int,
                    help='hidden dimension')
parser.add_argument('--lr', default=0.001, type=float,
                    help='learning rate')
parser.add_argument('--mode', type=str, choices=['e', 't'], nargs="+",
                    help='runing mode, \'t\':train, \'e\':evaluation')
parser.add_argument('-p', default=False, action='store_true',
                    help='prediction')

def main(argv):
    args = parser.parse_args(argv[1:])
    ratings, movies, users = load_lm_1m_data()
    train_data, test_data = pre_excute(ratings)

    params = {
        'lambda': 10,
        'user_num': len(users),
        'item_num': len(movies),
        'reg': 0.05,
        'device': '/cpu:0',
        'dim': args.dim,
        'learning_rate': args.learning_rate
    }

    classifier = tf.estimator.Estimator(
        model_fn=train_fn,
        params=params,
        model_dir='./model')

    tensors_to_log = {
        'inferid': 'inferid',
        'label': 'label'
    }

    logging_hook=tf_debug.LocalCLIDebugHook()

    if 't' in args.mode:
        classifier.train(input_fn=lambda: train_fn(
            train_data.loc[:,['user','item']], train_data['ratings']),steps=1000,hooks=[logging_hook])

    if 'e' in args.mode:
        eval_result = classifier.evaluate(
             input_fn=lambda:eval_fn(test_data.loc[:,['user','item']], test_data['rate']))
        for i,thing in enumerate(eval_result):
              print(thing,":",eval_result[thing])
        print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    if 'p' in args.mode:
        predictions = classifier.predict(
            input_fn=lambda: pred_fn(test_data.iloc[232:290, :2]))
        expected = test_data.iloc[232:290, 2]

        for pred, rate in zip(predictions, expected):
            print("predict:",pred,"real rate:",rate)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)