# -*- coding: utf-8 -*-
# @Author  : hooker5427

import os
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras


train_file = "./data/titanic/train.csv"
eval_file = "./data/titanic/eval.csv"

train_df = pd.read_csv(train_file)
eval_df = pd.read_csv(eval_file)

y_train = train_df.pop('survived')
y_eval = eval_df.pop('survived')

#(627, 9) (264, 9) (627,) (264,)
print ( train_df.shape , eval_df.shape ,  y_train.shape , y_eval.shape )

categorical_columns = ['sex', 'n_siblings_spouses', 'parch', 'class',
                       'deck', 'embark_town', 'alone']
numeric_columns = ['age', 'fare']

feature_columns = []
for categorical_column in categorical_columns:
    vocab = train_df[categorical_column].unique()
    print(categorical_column, vocab)
    feature_columns.append(
        tf.feature_column.indicator_column(
            tf.feature_column.categorical_column_with_vocabulary_list(
                categorical_column, vocab)))

for categorical_column in numeric_columns:
    feature_columns.append(
        tf.feature_column.numeric_column(
            categorical_column, dtype=tf.float32))


def make_dataset(data_df, label_df, epochs=10, shuffle=True,batch_size= 32 ):
    dataset = tf.data.Dataset.from_tensor_slices(  (dict(data_df),  label_df))
    if shuffle:
        dataset = dataset.shuffle(10000)
    dataset = dataset.repeat(epochs).batch(batch_size)
    return dataset



import os 
output_dir = 'baseline_model_1'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

baseline_estimator = tf.estimator.BaselineClassifier(model_dir = output_dir,n_classes = 2)
baseline_estimator.train(input_fn = lambda : make_dataset(train_df, y_train, epochs = 100) )
baseline_estimator.evaluate(input_fn = lambda : make_dataset(eval_df, y_eval, epochs = 1, shuffle = False, batch_size = 20))

# linear_model
linear_output_dir = 'linear_model'
if not os.path.exists(linear_output_dir):
    os.mkdir(linear_output_dir)
linear_estimator = tf.estimator.LinearClassifier(
                                            model_dir = linear_output_dir,
                                            n_classes = 2,
                                            feature_columns = feature_columns)
linear_estimator.train(input_fn = lambda : make_dataset(train_df, y_train, epochs = 100))
linear_estimator.evaluate(input_fn = lambda : make_dataset(eval_df, y_eval, epochs = 1, shuffle = False))


# dnn_model
dnn_output_dir = './dnn_model'
if not os.path.exists(dnn_output_dir):
    os.mkdir(dnn_output_dir)
dnn_estimator = tf.estimator.DNNClassifier(
                                        model_dir = dnn_output_dir,
                                        n_classes = 2,
                                        feature_columns=feature_columns,
                                        hidden_units = [128, 128],
                                        activation_fn = tf.nn.relu,
                                        optimizer = 'Adam')
dnn_estimator.train(input_fn = lambda : make_dataset(train_df, y_train, epochs = 100))
dnn_estimator.evaluate(input_fn = lambda : make_dataset(eval_df, y_eval, epochs = 1, shuffle = False))