import os
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.python.framework.ops import disable_eager_execution
from collections import Counter
# import sqlite3

# import custom dependencies
import Summarizer
import summarizer_data_utils
import summarizer_model_utils

# the dataset consists of 3 .csv files. we will concatenate them.
data = pd.read_csv('./articles1.csv',encoding='utf-8')
data1 = pd.read_csv('./articles2.csv',encoding='utf-8')
data2 = pd.read_csv('./articles3.csv',encoding='utf-8')
data = pd.concat([data, data1, data2])
data = data[data.publication != 'Breitbart']

# drop those.
data.dropna(subset=['title'], inplace = True)
# to make the transition from the amazon review example to this one as comfortable as possbile we just rename
# the columns.
data.rename(index = str, columns = {'title':'Summary', 'content':'Text'}, inplace = True)
data = data[['Summary', 'Text']]

# again we will not use all of the examples, but only pick some.
len_summaries = [len(summary) for i, summary in enumerate(data.Summary)]
len_texts = [len(text) for text in data.Text]

len_summaries_counted = Counter(len_summaries).most_common()
len_texts_counted = Counter(len_texts).most_common()
#len_summaries_counted[:10], len_texts_counted[:10]


# we will only use shorter texts, as I have limited resources and those are easier to learn
indices = [ind for ind, text in enumerate(data.Text) if 50 < len(text) < 200]
texts_unprocessed = data.Text[indices]
summaries_unprocessed = data.Summary[indices]
# articles from nyt and breitbart seem to have those endings, therefore
# we will remove those, as that is not relevant.
to_remove = ['- The New York Times', '- Breitbart']

summaries_unprocessed_clean = []
texts_unprocessed_clean = []

removed = 0
append = True
for sentence in summaries_unprocessed:
    append = True
    for r in to_remove:
        if sentence.endswith(r):
            sentence = sentence.replace(r, '.')
            summaries_unprocessed_clean.append(sentence.replace(r, '.'))
            removed+=1
            append = False
            break

    if append:
        summaries_unprocessed_clean.append(sentence)

# preprocess the texts and summaries.
# we have the option to keep_most or not. in this case we do not want 'to keep most', i.e. we will only keep
# letters and numbers.
# (to improve the model, this preprocessing step should be refined)
processed_texts, processed_summaries, words_counted = summarizer_data_utils.preprocess_texts_and_summaries(
    texts_unprocessed,
    summaries_unprocessed_clean,
    keep_most=False)

# some of the texts are empty remove those.
processed_texts_clean = []
processed_summaries_clean = []

for t, s in zip(processed_texts, processed_summaries):
    if t != [] and s != []:
        processed_texts_clean.append(t)
        processed_summaries_clean.append(s)

# create lookup dicts.
# most oft the words only appear only once.
# min_occureces set to 2 reduces our vocabulary by more than half.
specials = ["<EOS>", "<SOS>","<PAD>","<UNK>"]
word2ind, ind2word,  missing_words = summarizer_data_utils.create_word_inds_dicts(words_counted,specials = specials,min_occurences = 2)


# glove_embeddings_path = '/Users/thomas/Jupyter_Notebooks/Pro Deep Learning with Tensorflow/Notebooks/glove/glove.6B.300d.txt'
# embedding_matrix_save_path = './embeddings/my_embedding.npy'
# emb = summarizer_data_utils.create_and_save_embedding_matrix(word2ind,glove_embeddings_path,embedding_matrix_save_path)

# the embeddings from tf.hub.
# embed = hub.Module("https://tfhub.dev/google/nnlm-en-dim128/1")
embed = hub.KerasLayer("https://tfhub.dev/google/Wiki-words-250/1")
emb = embed(tf.convert_to_tensor([key for key in word2ind.keys()]))


with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    sess.run(tf.compat.v1.tables_initializer())
    disable_eager_execution()
    embedding = sess.run(emb)

np.save('./tf_hub_embedding_headlines.npy', embedding)

# converts words in texts and summaries to indices
converted_texts, unknown_words_in_texts = summarizer_data_utils.convert_to_inds(processed_texts_clean,word2ind,eos = False)
converted_summaries, unknown_words_in_summaries = summarizer_data_utils.convert_to_inds(processed_summaries_clean,word2ind,eos = True,sos = True)

## TRAINING ##
num_layers_encoder = 4
num_layers_decoder = 4
rnn_size_encoder = 300
rnn_size_decoder = 300

batch_size = 32
epochs = 150
clip = 20
keep_probability = 0.8
learning_rate = 0.0005
max_lr=0.005
learning_rate_decay_steps = 100
learning_rate_decay = 0.90


pretrained_embeddings_path = './tf_hub_embedding_headlines.npy'
summary_dir = os.path.join('./tensorboard/headlines')

use_cyclic_lr = True
inference_targets=True

## build graph and train the model
summarizer_model_utils.reset_graph()
summarizer = Summarizer.Summarizer(word2ind,
                                   ind2word,
                                   save_path='./models/headlines/my_model',
                                   mode='TRAIN',
                                   num_layers_encoder = num_layers_encoder,
                                   num_layers_decoder = num_layers_decoder,
                                   rnn_size_encoder = rnn_size_encoder,
                                   rnn_size_decoder = rnn_size_decoder,
                                   batch_size = batch_size,
                                   clip = clip,
                                   keep_probability = keep_probability,
                                   learning_rate = learning_rate,
                                   max_lr=max_lr,
                                   learning_rate_decay_steps = learning_rate_decay_steps,
                                   learning_rate_decay = learning_rate_decay,
                                   epochs = epochs,
                                   pretrained_embeddings_path = pretrained_embeddings_path,
                                   use_cyclic_lr = use_cyclic_lr,)
#                                    summary_dir = summary_dir)

summarizer.build_graph()
summarizer.train(converted_texts,
                 converted_summaries)

# inference
summarizer_model_utils.reset_graph()
summarizer = Summarizer.Summarizer(word2ind,
                                   ind2word,
                                   './models/headlines/my_model',
                                   'INFER',
                                   num_layers_encoder = num_layers_encoder,
                                   num_layers_decoder = num_layers_decoder,
                                   batch_size = len(converted_texts[:50]),
                                   clip = clip,
                                   keep_probability = 1.0,
                                   learning_rate = 0.0,
                                   beam_width = 5,
                                   rnn_size_encoder = rnn_size_encoder,
                                   rnn_size_decoder = rnn_size_decoder,
                                   inference_targets = False,
                                   pretrained_embeddings_path = pretrained_embeddings_path)

summarizer.build_graph()
preds = summarizer.infer(converted_texts[:50],
                         restore_path =  './models/headlines/my_model',
                         targets = converted_summaries[:50])

# show results
print(summarizer_model_utils.sample_results(preds,
                                      ind2word,
                                      word2ind,
                                      converted_summaries[:50],
                                      converted_texts[:50]))