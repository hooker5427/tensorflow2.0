# -*- coding: utf-8 -*-
# @Author  : hooker5427

import os
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras


train_file = "./data/titanic/clean_train.csv"

train_df_all = pd.read_csv(train_file)
train_df_all , y_train = train_df_all , train_df_all.pop("Survived")
train_df = train_df_all.iloc[:600 , :] 
eval_df = train_df_all.iloc[600: ]


y_train ,y_eval  = y_train[:600] , y_train[600:]


categorical_columns = ['Sex', 'Ticket',"Parch" , 'Pclass' , 'Embarked' ]
numeric_columns = ['Age', 'Fare' ,'SibSp' ,'average_Fare' ,'family_size']

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

feature_columns.append(tf.feature_column.indicator_column(
    tf.feature_column.crossed_column(
            ['Age', 'Sex'], hash_bucket_size = 100)))

feature_columns.append(tf.feature_column.indicator_column(
    tf.feature_column.crossed_column(
            ['Fare', 'Parch'], hash_bucket_size = 100)))

feature_columns.append(tf.feature_column.indicator_column(
    tf.feature_column.crossed_column(
            ['Embarked', 'Ticket'], hash_bucket_size = 100)))


def make_dataset(data_df, label_df, epochs=10, shuffle=True,batch_size= 32 ):
    dataset = tf.data.Dataset.from_tensor_slices(  (dict(data_df),  label_df))
    if shuffle:
        dataset = dataset.shuffle(10000)
    dataset = dataset.repeat(epochs).batch(batch_size)
    return dataset

model = keras.models.Sequential([
    keras.layers.DenseFeatures(feature_columns),
    keras.layers.Dense(500, activation='relu'),
    keras.layers.BatchNormalization() , 
    keras.layers.Dropout( 0.2 ) , 
    keras.layers.Dense(500, activation='relu'),
    keras.layers.BatchNormalization() , 
    keras.layers.Dropout( 0.5 ) , 
    keras.layers.Dense(300, activation='relu'),
    keras.layers.Dense(2, activation='softmax'),
])
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=keras.optimizers.Adam(lr=0.01),
              metrics=['accuracy'])

epochs=100 
batch_size = 64
train_dataset = make_dataset(train_df, y_train, epochs =epochs  )
eval_dataset = make_dataset(eval_df, y_eval, epochs = 1, shuffle = False)
history = model.fit(train_dataset,
                    validation_data = eval_dataset,
                    steps_per_epoch =  train_df.shape[0] //  batch_size ,
                    validation_steps = 8,
                    epochs = epochs  )


def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 3 )
    plt.show()

plot_learning_curves(history)

