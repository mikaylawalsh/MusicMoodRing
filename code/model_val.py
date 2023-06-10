from json.encoder import INFINITY
from preprocess_val import get_data
import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa
# imports for plotting
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


class Model(tf.keras.Model):
    def __init__(self):
        """
        Initializes the Model instance with values for model hyperparameters, embedding layer, 
        convolutional layer, max pooling layer, LSTM layer, dropout layer, flatten layer, 
        and sequential layer. Also initializes the optimizer and loss function as well as 
        dataframes for model visualization.
        """
        super(Model, self).__init__()

        # for model
        self.batch_size = 32
        self.num_classes = 1  # only predicting one value
        self.lr = .001
        self.epochs = 60
        # self.stride = (default is 1 so only need this if want something different?)
        self.padding = "SAME"
        self.embedding_size = 100
        self.vocab_size = 96272
        self.hidden_size = 128
        self.momentum = 0.9  # used in optimzer

        # for plots
        self.epoch_list = []
        self.test_list = []
        self.plot_df_train = pd.DataFrame()
        self.plot_df_test = pd.DataFrame()

        self.embedding = tf.keras.layers.Embedding(
            self.vocab_size, self.embedding_size, mask_zero=True)

        self.permute = tf.keras.layers.Permute((2, 1), input_shape=(529, 64))

        self.conv1d = tf.keras.layers.Conv1D(
            16, 2, strides=2, padding=self.padding, activation="relu", kernel_initializer="HeNormal")

        self.permute2 = tf.keras.layers.Permute(
            (1, 2), input_shape=(529, 64))

        # LSTM or #GRU --> can use either but we ultimately
        # found LSTM to have better results
        self.LSTM = tf.keras.layers.LSTM(100)
        # self.GRU = tf.keras.layers.GRU(100)

        self.drop = tf.keras.layers.Dropout(.5)
        self.flat = tf.keras.layers.Flatten()

        # sigmoid activation for values from 0 to 1 (constraint given in dataset)
        self.seq = tf.keras.Sequential([tf.keras.layers.Dense(
            128, activation="relu"), tf.keras.layers.Dropout(.5),
            tf.keras.layers.Dense(self.num_classes, activation="sigmoid")])

        self.optimizer = tf.keras.optimizers.SGD(self.lr, self.momentum)
        self.loss = tf.keras.losses.MeanSquaredError()

    def call(self, inputs):
        """
        Passes the input tensor through the model's layers and returns the output logits.

        Args:
        - inputs: the input tensor to be passed through the model's layers

        Returns:
        - logits: the output logits of the model after layers
        """

        logits = self.embedding(inputs)
        logits = self.conv1d(logits)
        logits = tf.nn.max_pool(logits, 2, strides=None, padding=self.padding)
        logits = self.LSTM(logits)
        # logits = self.GRU(logits) # uncomment if using GRU
        logits = self.flat(logits)
        logits = self.drop(logits)
        logits = self.seq(logits)

        return logits

    def r2_score(self, logits, labels):
        """
        Calculates the R-squared metric for the model's predictions.
        R_squared served as out accuracy for this model.
        Higher r_squared scores indicated more optimal/better trained results.

        Args:
        - logits: the model's output logits
        - labels: the ground truth labels

        Returns:
        - result: the R-squared metric value
        """

        metric = tfa.metrics.r_square.RSquare()
        metric.update_state(labels, logits)
        result = metric.result()

        return result.numpy()


def train(model, train_lyrics, train_labels):
    """
    Trains the model for one epoch on the given training inputs and labels. 
    Uses forwards and backwards pass.

    Args:
    - model: the model to be trained
    - train_lyrics: the lyrics to be used as inputs for training the model
    - train_labels: the corresponding labels for the training lyrics

    Returns:
    - avg_loss: the average loss of the model on the training data
    - avg_r2: the average r2 score of the model on the training data
    """

    avg_r2 = 0
    avg_loss = 0
    counter = 0

    index_range = tf.random.shuffle(range(len(train_lyrics)))
    shuffled_lyrics = tf.gather(train_lyrics, index_range)
    shuffled_labels = tf.gather(train_labels, index_range)

    # train_lyrics.shape[0] + 1 if tensor
    # batch data and train each batch
    for batch_num, b1 in enumerate(range(model.batch_size, len(train_lyrics) + 1, model.batch_size)):
        b0 = b1 - model.batch_size
        batch_lyrics = shuffled_lyrics[b0:b1]
        batch_labels = shuffled_labels[b0:b1]

        with tf.GradientTape() as tape:
            logits = model(batch_lyrics)

            loss = model.loss(batch_labels, logits)

            grads = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(
                zip(grads, model.trainable_variables))
        r_squared = model.r2_score(logits, batch_labels)

        avg_r2 += r_squared
        avg_loss += loss
        counter += 1

        # for plots/charts
        model.epoch_list.append((r_squared, loss))
        model.plot_df_train = pd.DataFrame(
            model.epoch_list, columns=['r_squared', 'loss'])

        # shows training progress
        print(
            f"\r[Train {batch_num+1:4n}/{3764}]\t loss={loss:.3f}\t r_squared: {r_squared:.3f}", end='')
    print()

    return avg_loss/counter, avg_r2/counter


def test(model, test_lyrics, test_labels):
    """
    Tests the model on the test data and calculates the average R-squared and loss based on valence scores.

    Args:
    - model: the trained model to be tested
    - test_lyrics: a numpy array of lyrics to be used as inputs for testing the model
    - test_labels: a numpy array of corresponding valence scores for the test lyrics

    Returns:
    - avg_r2: the average R-squared of the model on the test data
    - avg_loss: the average loss of the model on the test data
    """

    avg_r2 = 0
    avg_loss = 0
    counter = 0
    for batch_num, b1 in enumerate(range(model.batch_size, len(test_lyrics) + 1, model.batch_size)):
        b0 = b1 - model.batch_size
        batch_lyrics = test_lyrics[b0:b1]
        batch_labels = test_labels[b0:b1]

        logits = model(batch_lyrics)
        loss = model.loss(batch_labels, logits)

        r2 = model.r2_score(logits, batch_labels)
        avg_r2 += r2
        avg_loss += loss
        counter += 1

        model.test_list.append((r2, loss))
        model.plot_df_test = pd.DataFrame(
            model.test_list, columns=['r_squared', 'loss'])

        # shows testing progress
        print(
            f"\r[Valid {batch_num+1:4n}/{941}]\t loss={loss:.3f}\t r_squared: {r2:.3f}", end='')

    print()
    return avg_r2/(test_lyrics.shape[0]/model.batch_size), avg_loss/(test_lyrics.shape[0]/model.batch_size)


# functions for charts
def plot_results_train(plot_df: pd.DataFrame) -> None:
    '''uses r_squared and loss inputs to graph loss vs. r_squared training results'''
    plot_df.plot.scatter(x='r_squared', y='loss',
                         title="r_squared results training")


def plot_results_test(plot_df: pd.DataFrame) -> None:
    '''uses r_squared and loss inputs to graph loss vs. r_squared testing results'''
    plot_df.plot.scatter(x='r_squared', y='loss',
                         title="r_squared results testing")


def main():

    train_lyrics, test_lyrics, train_labels, test_labels = get_data(
        "data/labeled_lyrics_cleaned.csv")

    model = Model()

    # loops through epochs
    for e in range(model.epochs):
        print("epoch", e+1)
        train(model, train_lyrics, train_labels)

    t = test(model, test_lyrics, test_labels)

    tf.print("Final R2 Score:", t[0])

    plot_results_train(model.plot_df_train)
    plt.show()

    plot_results_test(model.plot_df_test)
    plt.show()

    return


if __name__ == '__main__':
    main()
