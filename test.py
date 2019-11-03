import tensorflow as tf
from helper import *
import io

path = "./data/en-de.txt"
source_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
target_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')

# creating tokenizer
lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
word_pairs = [[preprocess_sentence(w) for w in l.split('\t')] for l in lines[:100]]
inp_text, target_text = zip(*word_pairs)
source_tokenizer.fit_on_texts(inp_text)
target_tokenizer.fit_on_texts(target_text)
print("size of source tokenizer is {}", len(source_tokenizer.word_index))
print("size of target tokenizer is {}", len(target_tokenizer.word_index))

def line_process(lines):
    word_pairs = [[preprocess_sentence(w) for w in l.strip().split('\t')] for l in lines]
    source_text, target_text = zip(*word_pairs)


    # tensor = source_tokenizer.texts_to_sequences(source_text)
    # source_text = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')
    # tensor = target_tokenizer.texts_to_sequences(target_text)
    # target_text = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')

    return source_text, target_text


line_dataset = tf.data.TextLineDataset([path])
line_dataset = line_dataset.shuffle(1000).batch(3)
iter = line_dataset.make_one_shot_iterator()
with tf.Session() as sess:
    lines = iter.get_next()
    line_split = tf.string_split(lines, '\t')
    print(lines)
    print(line_split)
    print(line_split.values)
    source_text, target_text = tf.py_func(line_process, [lines], [tf.string, tf.string])
    lines, source_text, line_split_valus = sess.run([lines, source_text, line_split.values])
    # print(lines)
    print(source_text)
    # print(line_split_valus)
