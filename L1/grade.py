# -*- coding: utf-8 -*-
# @Author  : hooker5427

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pandas as pd
import os
import sys
import time
import tensorflow as tf

from tensorflow import keras




def f( x ) :
    return  x*5 + x**2 


def g( x1 , x2 ):
    return  x1**2 + x2**2
 
# 求一阶导数
def grade1( f, x , eps = 1e-3 ) :
    x_greater  = x + eps 
    x_less  = x- eps 
    return ( f(x_greater) - f(x_less)) /  ( 2.0* eps )



# 求二阶导数
def  grade2 ( g , x1 , x2  , eps = 1e-3) :
    idx1 =  grade1( lambda x : g (x  , x2 ) ,  x1 , eps  ) 
    idx2 =  grade1( lambda x : g( x1 , x ) , x2 , eps) 
    return idx1 ,idx2 



#使用tensorflow自动进行求导操作
x1 = tf.Variable(2.0)
x2 = tf.Variable(3.0)
with tf.GradientTape() as tape:
    z = g(x1, x2)

dz_x1 = tape.gradient(z, x1)
print(dz_x1.numpy())


# 默认只能求导一次 
x1 = tf.Variable(2.0)
x2 = tf.Variable(3.0)
with tf.GradientTape(persistent = True) as tape:
    z = g(x1, x2)

dz_x1 = tape.gradient(z, x1)
dz_x2 = tape.gradient(z, x2)
print(dz_x1, dz_x2)

del tape

x1 = tf.Variable(2.0)
x2 = tf.Variable(3.0)
with tf.GradientTape(persistent=True) as outer_tape:
    with tf.GradientTape(persistent=True) as inner_tape:
        z = g(x1, x2)
    inner_grads = inner_tape.gradient(z, [x1, x2])
    
outer_grads = [outer_tape.gradient(inner_grad, [x1, x2])
               for inner_grad in inner_grads]
print(outer_grads) 
del inner_tape
del outer_tape


learning_rate = 0.1
x = tf.Variable(0.0)

for _ in range(100):
    with tf.GradientTape() as tape:
        z = f(x)
    dz_dx = tape.gradient(z, x)
    x.assign_sub(learning_rate * dz_dx)
print(x)


learning_rate = 0.1
x = tf.Variable(0.0)

optimizer = keras.optimizers.SGD(lr = learning_rate)

for _ in range(100):
    with tf.GradientTape() as tape:
        z = f(x)
    dz_dx = tape.gradient(z, x)
    optimizer.apply_gradients([(dz_dx, x)])
print(x)