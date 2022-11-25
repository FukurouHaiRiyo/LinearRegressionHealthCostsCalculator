import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.layers.experimental import preprocessing
from sklearn.model_selection import train_test_split

PATH = 'path/to/insurance.csv'

dataset = pd.read_csv(PATH)
print(dataset.tail())

sex = pd.get_dummies(dataset['sex'])
smoker = pd.get_dummies(dataset['smoker'])
region = pd.get_dummies(dataset['region'])
dataset = pd.concat([dataset, sex, smoker, region], axis=1)
dataset = dataset.drop(['sex', 'smoker', 'region'], axis=1)

train = dataset.sample(frac=0.8, random_state=0)
test = dataset.drop(train.index)

train_labels = train.pop('expenses')
test_labels = test.pop('expenses')

normalizer = preprocessing.Normalization()
normalizer.adapt(np.array(train))

model = tf.keras.Sequential([
    normalizer,
    layers.Dense(units=1)
])

model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')

history = model.fit(
    train, train_labels, 
    epochs=100,
    # suppress logging
    verbose=2,
    # Calculate validation results on 20% of the training data
    validation_split = 0.2)

#evaluating the model
test_results = {}
pred = model.predict(test).flatten()
test_results['linear_model'] = model.evaluate(
      test, test_labels, verbose=0
)

plt.scatter(test_labels, pred)
plt.xlabel('True Values [expenses]')
plt.ylabel('Predictions [expenses]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0, plt.xlim()[1]])
plt.ylim([0, plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])
plt.show()
