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

dim = 15
EPOCH_MAX = 100
reg = 0.05

def model_fn(
        features,  # This is batch_features from input_fn
        mode,
        params,
        labels=None,  # This is batch_labels from input_fn
):  # And instance of tf.estimator.ModeKeys, see below

    user_batch = features['users']
    item_batch = features['items']
    with tf.device("/cpu:0"):
        bias_global = tf.get_variable("bias_global", shape=[])
        w_bias_user = tf.get_variable("embd_bias_user", shape=[params['user_num']])
        w_bias_item = tf.get_variable("embd_bias_item", shape=[params['item_num']])
        bias_user = tf.nn.embedding_lookup(w_bias_user, user_batch, name="bias_user")
        bias_item = tf.nn.embedding_lookup(w_bias_item, item_batch, name="bias_item")
        w_user = tf.get_variable("embd_user", shape=[params['user_num'], dim],
                                 initializer=tf.truncated_normal_initializer(stddev=0.02))
        w_item = tf.get_variable("embd_item", shape=[params['item_num'], dim],
                                 initializer=tf.truncated_normal_initializer(stddev=0.02))
        embd_user = tf.nn.embedding_lookup(w_user, user_batch, name="embedding_user")
        embd_item = tf.nn.embedding_lookup(w_item, item_batch, name="embedding_item")

        infer = tf.reduce_sum(tf.multiply(embd_user, embd_item), 1)
        infer = tf.add(infer, bias_global)
        infer = tf.add(infer, bias_user)
        infer = tf.add(infer, bias_item, name="svd_inference")

        regularizer = tf.add(tf.nn.l2_loss(embd_user), tf.nn.l2_loss(embd_item), name="svd_regularizer")

        prediction = tf.add(tf.cast(infer,tf.int8),1)

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode, predictions=prediction)

        cost_l2 = tf.nn.l2_loss(tf.subtract(infer, labels))
        penalty = tf.constant(reg, dtype=tf.float32, shape=[], name="l2")
        cost = tf.add(cost_l2, tf.multiply(regularizer, penalty))

        global_step = tf.train.get_global_step()
        assert global_step is not None
        learning_rate = tf.train.exponential_decay(params['learning_rate'],
                                                   global_step,
                                                   decay_steps=500,
                                                   decay_rate=0.9)
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost, global_step=global_step)

        accurmid = tf.metrics.accuracy(labels,tf.cast(infer,tf.int8))
        metrics = {'accuracy': accurmid}
        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode,
                loss=cost,
                eval_metric_ops=metrics)

        return tf.estimator.EstimatorSpec(
            mode,
            loss=cost,
            train_op=train_op
        )