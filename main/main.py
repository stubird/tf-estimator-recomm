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
from models.svd import *
import argparse
import random
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=1000, type=int,
                    help='batch size')
parser.add_argument('--train_steps', default=200000, type=int,
                    help='number of training steps')
parser.add_argument('--datafile', default="", type=str,
                    help='path of data')
parser.add_argument('--dim', default=15, type=int,
                    help='hidden dimension')
parser.add_argument('--lr', default=0.0001, type=float,
                    help='learning rate')
parser.add_argument('-t', default=False, action='store_true',
                    help='running in training mode')
parser.add_argument('-p', default=False, action='store_true',
                    help='running in predicting mode')
parser.add_argument('-e', default=False, action='store_true',
                    help='running in evaluating mode')
parser.add_argument('--debug', default=False, action='store_true',
                    help='debug mode')


def main(argv):
    args = parser.parse_args(argv[1:])
    ratings, movies, users = load_lm_1m_data()
    train_data, test_data = pre_excute(ratings)

    params = {
        'lambda': 10,
        'user_num': users['index'].max(),
        'item_num': movies['index'].max(),
        'reg': 0.05,
        'device': '/cpu:0',
        'dim': args.dim,
        'learning_rate':args.lr
    }

    classifier = tf.estimator.Estimator(
        model_fn=model_fn,
        params=params,
        model_dir='./model')

    if args.t:
        if args.debug:
            logging_hook = tf_debug.LocalCLIDebugHook()
            classifier.train(input_fn=lambda: train_fn(
                train_data.loc[:,['users','items']], train_data['ratings']),steps=args.train_steps,hooks=[logging_hook])
        else:
            classifier.train(input_fn=lambda: train_fn(
                train_data.loc[:, ['users', 'items']], train_data['ratings']), steps=args.train_steps)

    if args.e:
        eval_result = classifier.evaluate(
             input_fn=lambda:eval_fn(test_data.loc[:,['users','items']], test_data['ratings']))
        for i,thing in enumerate(eval_result):
              print(thing,":",eval_result[thing])

    if args.p:
        a = random.randint(0, len(users)-11)
        b = a + 10
        predictions = classifier.predict(
            input_fn=lambda: pred_fn(test_data.iloc[a:b, :2]))
        expected = test_data.iloc[a:b, 2]

        for pred, rate in zip(predictions, expected):
            print("user:{},predict:{},real rate:{}.".format(a,pred,rate))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)