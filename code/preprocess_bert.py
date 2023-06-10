import nltk
from audioop import avg
import tensorflow as tf
import numpy as np
import pickle
import csv
import pandas as pd
from functools import reduce
from nltk.corpus import stopwords
from transformers import BertTokenizer, BertModel


def get_data(file_path):
    data = pd.read_csv(file_path)

    lyrics = data['lyrics']
    labels = data['label']
    labels = [[i] for i in labels]

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    input_ids = []
    attention_masks = []
    for song in lyrics: 
        encoded_dict = tokenizer.encode_plus(song, add_special_tokens=True, max_length=50, pad_to_max_length=True, return_attention_mask=True, return_tensors='tf')
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
    
    input_ids_tens = tf.convert_to_tensor(input_ids)
    attention_masks_tens = tf.convert_to_tensor(attention_masks)

    embedding = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)

    final_embed = []
    tok_embed = []

    embedding.eval()

    for i in range(len(input_ids)):

        outputs = embedding(input_ids[i], attention_mask=attention_masks_tens)
        hidden_states = outputs[2]
        print("hiden_states", len(hidden_states), len(hidden_states[0]), len(hidden_states[0][0]), len(hidden_states[0][0][0])) # 13, 150.., 50, 768
        tok_embed[i] = tf.stack(hidden_states)

        print("tok_embed stack", tok_embed.shape)
        print("tok_embed[0] stack", tok_embed[0].shape)

        tok_embed[i] = tf.squeeze(tok_embed[i], axis=1)

        print("tok_embed squeeze", tok_embed.shape)
        print("tok_embed[0] squeeze", tok_embed[0].shape)

        tok_embed[i] = tf.transpose(tok_embed[i], perm=(1,0,2))
    
        print("tok_embed transpose", tok_embed.shape)
        print("tok_embed[0] transpose", tok_embed[0].shape)

        tok_vec_sum = []
        for tok in tok_embed: # tok_embed or tok_embed[i]
            sum_vec = tf.reduce_sum(tok[-4:], axis=0)
            tok_vec_sum.append(sum_vec)
        final_embed[i] = tok_vec_sum 

    print("final_embed", final_embed.shape)
    # split into test and train 

    return tf.convert_to_tensor(final_embed), tf.convert_to_tensor(labels)


def main():
    # can delete later -- just for testing

    X0, X1 = get_data(
        "data/labeled_lyrics_cleaned.csv")

    # print(X0)
    # print(Y0)
    # print(X1)
    # print(Y1)

    return


if __name__ == '__main__':
    main()

#transfer learing
#add layers on top of bert--> would change project quite a bit, might take up ram, use as alternative, try other transformers as well