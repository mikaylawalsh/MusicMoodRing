import nltk
from audioop import avg
import tensorflow as tf
import numpy as np
import pickle
import csv
import pandas as pd
from functools import reduce
from nltk.corpus import stopwords


def get_data(file_path):
    """
    Reads data from a csv file and preprocesses it for training and testing.

    Args:
    file_path (str): The file path to the csv file containing the data.

    Returns:
    Tuple of four tensors:
    - train_lyrics: Tensor containing the preprocessed training lyrics.
    - test_lyrics: Tensor containing the preprocessed testing lyrics.
    - train_labels: Tensor containing the training labels.
    - test_labels: Tensor containing the testing labels.
    """
    data = pd.read_csv(file_path)

    lyrics = data['lyrics']
    labels = data['label']

    # Puts each set of lyrics into own list
    # clean data
    stop_words = set(stopwords.words('english'))

    lyrics = [[word.lower().strip("!()-',.?*{};:¡\"“‘~…’—–”\\")
               for word in song.split() if word not in stop_words] for song in lyrics]

    mean = np.mean([len(song) for song in lyrics])
    std = np.std([len(song) for song in lyrics])

    upper_bound = mean + 2*std
    lower_bound = mean - 2*std

    # even out data inputs (by length)
    indices = np.nonzero([1 if len(song) <= upper_bound and len(
        song) >= lower_bound else 0 for song in lyrics])[0]

    lyrics = [lyrics[i] for i in indices]
    labels = [labels[i] for i in indices]

    for song in range(len(lyrics)):
        lyrics[song] = lyrics[song][:50]

    # If we wanted one list of all lyrics we could do it this way:
    # train_lyrics_list = []
    # for x in train_lyrics:
    #     train_lyrics_list.append(x.split())
    # test_lyrics_list = []
    # for y in test_lyrics:
    #     test_lyrics_list.append(y.split())

    # remove duplicate songs, cleans data some more
    unique = []
    for song in lyrics:
        unique.extend(song)
    unique = sorted(set(unique))

    vocabulary = {w: i for i, w in enumerate(unique, start=1)}

    lyrics = [list(map(lambda x: vocabulary[x], song))
              for song in lyrics]

    # pad songs for tensor conversion
    lyrics = tf.keras.preprocessing.sequence.pad_sequences(
        lyrics, padding='post')  # returns np array

    # labeled_lyrics_clean (math):
    # total = 1103, 80% = 882, 20% = 221
    # labeled_lyrics
    # total = 150568, 80% = 120,454, 20% = 30,114

    # batch data
    index_range = tf.random.shuffle(range(len(lyrics)))
    shuffled_lyrics = tf.gather(lyrics, index_range)
    shuffled_labels = tf.gather(labels, index_range)
    shuffled_labels = [[i] for i in shuffled_labels]

    train_lyrics, test_lyrics = shuffled_lyrics[:120454], shuffled_lyrics[120454:]
    train_labels, test_labels = shuffled_labels[:120454], shuffled_labels[120454:]

    return tf.convert_to_tensor(train_lyrics), tf.convert_to_tensor(test_lyrics), tf.convert_to_tensor(train_labels), tf.convert_to_tensor(test_labels)


def main():
    # this method is mainly used for testing
    X0, Y0, X1, Y1 = get_data(
        "data/labeled_lyrics_cleaned.csv")

    return


if __name__ == '__main__':
    main()
