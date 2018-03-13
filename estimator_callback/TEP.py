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

"""
call back for Train, Evaluation and Prediction
"""

def eval_fn(feature, rate, batch_size = 10):
    """Evaluation call back

    Args:
      feature: A DateFrame with `user` and `item`
      rate: A DateFrame with `rate`
      batch_szie: batch size

    Returns:
      A nested structure of `tf.Tensor` objects.
    """
    dataset = tf.data.Dataset.from_tensor_slices((dict(feature), rate)).batch(batch_size)

    iterator = dataset.make_one_shot_iterator()
    batch_feature, batch_rate = iterator.get_next()
    return batch_feature, batch_rate

def pred_fn(feature):
    """Prediction call back

    Args:
      feature: A DateFrame with `user` and `item`

    Returns:
      A nested structure of `tf.Tensor` objects.
    """

    dataset = tf.data.Dataset.from_tensor_slices(dict(feature)).batch(1)

    iterator = dataset.make_one_shot_iterator()
    batch_feature = iterator.get_next()
    return batch_feature

def train_fn(feature, rate, batch_size = 10000,repeat_count=None, shuffle_count=1):
    """Train call back

    Args:
      feature: A DateFrame with `user` and `item`
      rate: A DateFrame with `rate`
      batch_szie: batch size
      repeat_count: repeat count of data
      shuffle_count: shuffle count

    Returns:
      A nested structure of `tf.Tensor` objects.
    """

    dataset = (tf.data.Dataset.from_tensor_slices((dict(feature),rate))
        #.skip(1)  # Skip header row
        #.map(decode_csv, num_parallel_calls=4)  # Decode each line
        #.cache() # Warning: Caches entire dataset, can cause out of memory
        .shuffle(shuffle_count)  # Randomize elems (1 == no operation)
        .repeat(repeat_count)    # Repeats dataset this # times
        .batch(batch_size)
        #.prefetch(1)  # Make sure you always have 1 batch ready to serve
    )

    iterator = dataset.make_one_shot_iterator()
    batch_feature, batch_rate = iterator.get_next()
    return batch_feature, batch_rate