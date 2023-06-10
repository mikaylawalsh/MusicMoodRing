from json.encoder import INFINITY
from preprocess_mood import get_data
import tensorflow as tf
import numpy as np
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
        self.num_classes = 3
        self.lr = .001
        self.epochs = 50
        self.momentum = 0.9
        self.padding = "SAME"
        self.embedding_size = 100
        self.vocab_size = 15245
        self.hidden_size = 64
        # for plot
        self.epoch_list = []
        self.plot_df = pd.DataFrame()
        self.plot_class_df = pd.DataFrame()
        self.plot_sadness = pd.DataFrame()
        self.plot_tenderness = pd.DataFrame()
        self.plot_tension = pd.DataFrame()

        self.embedding = tf.keras.layers.Embedding(
            self.vocab_size, self.embedding_size)

        self.permute = tf.keras.layers.Permute((2, 1), input_shape=(529, 64))
        self.conv1d = tf.keras.layers.Conv1D(
            16, 2, strides=2, activation='relu')

        self.maxpool = tf.keras.layers.MaxPool1D(2)
        self.LSTM = tf.keras.layers.LSTM(
            40, activation='leaky_relu')
        self.drop = tf.keras.layers.Dropout(.5)
        self.flat = tf.keras.layers.Flatten()
        self.seq = tf.keras.Sequential([tf.keras.layers.Dense(
            64, kernel_regularizer=tf.keras.regularizers.L1(.01), activation='relu'), tf.keras.layers.Dropout(.5), tf.keras.layers.Dense(self.num_classes, activation='softmax')])

        # stochastic gradient descent optimizer
        self.optimizer = tf.keras.optimizers.SGD(self.lr, self.momentum)
        self.loss = tf.keras.losses.CategoricalCrossentropy()

    def call(self, inputs):
        """
        Defines the forward pass of the Model by applying each layer in sequence
        to the input tensor.

        Args:
        inputs: A tensor representing the input sequences of the songs.

        Returns:
        logits: A tensor representing the probabilities of mood based on lyrics.
        """

        # model
        logits = self.embedding(inputs)
        logits = self.conv1d(logits)
        logits = self.maxpool(logits)
        logits = self.LSTM(logits)
        logits = self.flat(logits)
        logits = self.drop(logits)
        logits = self.seq(logits)

        return logits

    def accuracy(self, logits, labels):
        """
        Computes the accuracy of the Model based on number of correctly guessed classes
        using one hot encoded label vectors.

        Args:
        logits (tf.Tensor): A tensor representing the predicted logits.
        labels (tf.Tensor): A tensor representing the true labels.

        Returns:
        accuracy (float): The computed accuracy of the Model.
        """

        num_correct_classes = 0
        for song in range(self.batch_size):
            if tf.argmax(logits[song]) == tf.argmax(labels[song]):
                num_correct_classes += 1
        accuracy = num_correct_classes/self.batch_size
        return accuracy

    def acc_per_class(self, logits, labels):
        """
        Computes the accuracy of the Model for each class looking at the accuracy distribution
        across the different mood labels: sadness, tension, and tenderness.
        Used mainly for testing and graphing purposes.

        Args:
        logits (tf.Tensor): A tensor representing the predicted logits.
        labels (tf.Tensor): A tensor representing the true labels.

        Returns:
        acc_tension (float): The computed accuracy of the Model for the 'tension' class.
        acc_sadness (float): The computed accuracy of the Model for the 'sadness' class.
        acc_tenderness (float): The computed accuracy of the Model for the 'tenderness' class.
        """
        correct_tension = 0
        tot_tension = 0
        correct_sadness = 0
        tot_sad = 0
        correct_tenderness = 0
        tot_tender = 0
        for song in range(self.batch_size):
            if tf.argmax(logits[song], axis=-1) == tf.argmax(labels[song]):
                if tf.argmax(labels[song]).numpy() == 1:
                    correct_tension += 1
                    tot_tension += 1
                elif tf.argmax(labels[song]).numpy() == 0:
                    correct_sadness += 1
                    tot_sad += 1
                else:
                    tot_tender += 1
                    correct_tenderness += 1
            else:
                if tf.argmax(labels[song]).numpy() == 1:
                    tot_tension += 1
                elif tf.argmax(labels[song]).numpy() == 0:
                    tot_sad += 1
                else:
                    tot_tender += 1

        # handling 0 case
        if tot_tension == 0:
            acc_tension = 0
        else:
            acc_tension = correct_tension/tot_tension
        if tot_sad == 0:
            acc_sadness = 0
        else:
            acc_sadness = correct_sadness/tot_sad
        if tot_tension == 0:
            acc_tension = 0
        else:
            acc_tenderness = correct_tenderness/tot_tender

        return acc_tension, acc_sadness, acc_tenderness

    def loss(self, labels, logits):
        """
        Computes the loss of the Model given the true labels and predicted logits.
        The model penializes wrong answers for sadness and praise correct answers for tension and tenderness.

        Args:
        labels (tf.Tensor): A tensor representing the true labels.
        logits (tf.Tensor): A tensor representing the predicted logits.

        Returns:
        loss (tf.Tensor): A tensor representing the computed loss of the Model.
        """
        # model has learned to always guess sadness so we made this more custom
        # loss function to handle that
        cce = tf.keras.losses.CategoricalCrossentropy()
        print(logits)
        for i in range(logits.shape[0]):
            copy = logits
            if tf.argmax(logits[i]) != tf.argmax(labels[i]):
                # make copy
                index = tf.argmax(logits[i]).numpy()
                copy = copy.numpy()
                copy[i][index] = .0001
                copy = tf.convert_to_tensor(logits)
            else:
                index = tf.argmax(logits[i]).numpy()
                copy = copy.numpy()
                copy[i][index] = .99
                copy = tf.convert_to_tensor(logits)
        # accounts for penalty or praise in categorical cross entropy calculation
        loss = cce(labels, copy)
        return loss


def train(model, train_lyrics, train_labels):
    """
    The method trains the model based on the training dataset and uses the forward and backward pass.
    It trains for one epoch. It shuffles the train_lyrics and train_labels, then splits the data into 
    batches and trains each batch. For each batch, it computes the loss, gradients, accuracy, and accuracy per class. 
    It also adds the accuracy and loss to the epoch list. After training, it sets up dataframes for charts/plots.
    
    Args:
    model: the model to be trained
    train_lyrics: the training lyrics as a tensor
    train_labels: the training labels as a tensor

    Returns: average loss and accuracy over all batches
    
    """
    # trains our model based on the training data set
    # uses forward and backward pass
    # trains for one epoch
    avg_acc = 0
    avg_loss = 0
    counter = 0

    index_range = tf.random.shuffle(range(len(train_lyrics)))
    shuffled_lyrics = tf.gather(train_lyrics, index_range)
    shuffled_labels = tf.gather(train_labels, index_range)

    # train_lyrics.shape[0] + 1 if tensor
    # splits into batches and trains each batch
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
        acc = model.accuracy(logits, batch_labels)

        # for chart and helpful print statements
        tension, sadness, tenderness = model.acc_per_class(
            logits, batch_labels)

        avg_acc += acc
        avg_loss += loss
        counter += 1

        model.epoch_list.append((acc, loss))

        # uncomment to see each individual label's score
        # print(f"\r[Train {batch_num+1}/{27}]\t tension: {tension:.3f}\t sadness: {sadness:.3f}\t tenderness: {tenderness:.3f}", end='')

        # shows training progress
        print(
            f"\r[Train {batch_num+1}/{27}]\t loss={loss:.3f}\t acc: {acc:.3f}", end='')
    print()

    # setting up dataframes for charts/plots
    model.plot_df = pd.DataFrame(
        model.epoch_list, columns=['accuracy', 'loss'])

    model.plot_class_df = pd.DataFrame({'classes': [
                                       'tension', 'sadness', 'tenderness'], 'accuracy': [tension, sadness, tenderness]})

    return avg_loss/counter, avg_acc/counter


def test(model, test_lyrics, test_labels):
    """
    Tests the model on the test inputs and labels.

    Args:
    - model: the trained model to be tested
    - test_lyrics: the lyrics to be used as inputs for testing the model
    - test_labels: the corresponding labels for the test lyrics

    Returns:
    - avg_acc: the average accuracy of the model on the test data
    - avg_loss: the average loss of the model on the test data
    """
    avg_acc = 0
    avg_loss = 0
    counter = 0

    # splits into batches and tests per batch
    for batch_num, b1 in enumerate(range(model.batch_size, len(test_lyrics) + 1, model.batch_size)):
        b0 = b1 - model.batch_size
        batch_lyrics = test_lyrics[b0:b1]
        batch_labels = test_labels[b0:b1]

        logits = model(batch_lyrics)
        loss = model.loss(batch_labels, logits)

        acc = model.accuracy(logits, batch_labels)
        avg_acc += acc
        avg_loss += loss
        counter += 1

        # shows testing progress
        print(
            f"\r[Valid {batch_num+1}/{6}]\t loss={loss:.3f}\t acc: {acc:.3f}", end='')

    print()
    return avg_acc/counter, avg_loss/counter


# loss vs. accuracy scatter plot
def plot_results(plot_df: pd.DataFrame) -> None:
    '''plots accuracy vs. loss for training data scatter plot'''
    plot_df.plot.scatter(x='accuracy', y='loss',
                         title="training accuracy results table")


# Following three functions show individual accuracy per class
def plot_sad(plot_df: pd.DataFrame) -> None:
    '''plots epoch vs. accuracy for sad data'''
    plot_df.plot.line(x='epoch', y='accuracy',
                      title="sad")


def plot_tender(plot_df: pd.DataFrame) -> None:
    '''plots epoch vs. accuracy for tenderness data'''
    plot_df.plot.line(x='epoch', y='accuracy',
                      title="tender")


def plot_tension(plot_df: pd.DataFrame) -> None:
    '''plots epoch vs. accuracy for tension data'''
    plot_df.plot.line(x='epoch', y='accuracy',
                      title="tension")


# bar chart for each class
def plot_classes(plot_class_df: pd.DataFrame) -> None:
    '''plots bar chart to compare the accuracy of each class'''
    plot_class_df.plot.bar(x='classes', y='accuracy',
                           title="accuracy per class")


def main():

    train_lyrics, test_lyrics, train_labels, test_labels = get_data(
        "data/singlelabel.csv")

    model = Model()

    sad = []
    tender = []
    tension_list = []

    # loops through epochs
    for e in range(model.epochs):
        print("epoch", e+1)
        train(model, train_lyrics, train_labels)

        #for charts
        sad.append((e, model.acc_per_class(train_lyrics, train_labels)[1]))
        tender.append((e, model.acc_per_class(train_lyrics, train_labels)[2]))
        tension_list.append(
            (e, model.acc_per_class(train_lyrics, train_labels)[0]))

    # for charts
    model.plot_sadness = pd.DataFrame(
        sad, columns=['epoch', 'accuracy'])
    model.plot_tenderness = pd.DataFrame(
        tender, columns=['epoch', 'accuracy'])
    model.plot_tension = pd.DataFrame(
        tension_list, columns=['epoch', 'accuracy'])

    t = test(model, test_lyrics, test_labels)

    tf.print("Final Accuracy:", t[0])

    plot_results(model.plot_df)
    plt.show()

    plot_classes(model.plot_class_df)
    plt.show()

    # This code goes with the accuracy for class charts
    # can uncomment to see but they are not super helpful given our dataset
    # plot_sad(model.plot_sadness)
    # plt.show()

    # plot_tension(model.plot_tension)
    # plt.show()

    # plot_tender(model.plot_tenderness)
    # plt.show()

    return


if __name__ == '__main__':
    main()
