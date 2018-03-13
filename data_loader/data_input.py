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
import os
import numpy as np
import pandas as pd
import tensorflow as tf

DATA_URL="http://files.grouplens.org/datasets/movielens/ml-1m.zip"

def maybe_download():
    #make dir `data`
    pwd = os.getcwd() + "\\data\\"
    if not os.path.exists(pwd):
        os.mkdir(pwd)
    tf.keras.utils.get_file(pwd+DATA_URL.split('/')[-1], DATA_URL,extract=True)
    return pwd

def load_lm_1m_data():
    data_path = maybe_download()

    data_struct = {
        'ratings':'ratings.dat',
        'movies':'movies.dat',
        'users':'users.dat'
    }

    ratings_column = ['users','items','ratings','timestamp']
    ratings = pd.read_csv(data_path+data_struct['ratings'], sep="::", names=ratings_column, header=0)

    movies_column = ['index', 'name', 'genre']
    movies = pd.read_csv(data_path + data_struct['movies'], sep="::", names=movies_column, header=0)

    users_column = ['index', 'gender', 'age', 'occupation', 'timestamp']
    users = pd.read_csv(data_path + data_struct['users'], sep="::", names=users_column, header=0)

    return ratings, movies, users

def pre_excute(ratings):
    ratings["user"] -= 1
    ratings["item"] -= 1
    for col in ("user", "item"):
        ratings[col] = ratings[col].astype(np.int32)
    ratings["ratings"] = ratings["ratings"].astype(np.float32)

    rows = len(ratings)
    df = ratings.iloc[np.random.permutation(rows)].reset_index(drop=True)

    split_index = int(rows * 0.9)
    df_train = df[0:split_index]
    df_test = df[split_index:].reset_index(drop=True)

    return df_train, df_test

