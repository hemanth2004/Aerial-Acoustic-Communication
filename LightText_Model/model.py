import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import tensorflow as tf
Dense = tf.keras.layers.Dense
Input = tf.keras.layers.Input
Sequential = tf.keras.Sequential
SparseCategoricalCrossEntropy = tf.keras.losses.SparseCategoricalCrossEntropy
regularizers = tf.keras.regularizers
optimizers = tf.keras.optimizers
sigmoid = tf.keras.activations.sigmoid
relu = tf.keras.activations.relu
linear = tf.keras.activations.linear

np.set_printoptions(precision=2)

print("hi")