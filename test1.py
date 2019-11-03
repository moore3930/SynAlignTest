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
    # line = line.strip().lower()
    line = [l.strip().lower() for l in lines]
    # source, target = zip(*line)
    return [line]

def line_process2(lines):
    # line = line.strip().lower()
    line = [l.strip().lower().split(b'\t') for l in lines]
    source_text, target_text = zip(*line)
    source_text_str = ['hello', 'world']
    print(type(source_text_str))
    source_ids = source_tokenizer.texts_to_sequences(source_text_str)
    # target_ids = target_tokenizer.texts_to_sequences(target_text)
    print(source_ids)
    return source_text, target_text

def line_process3(lines):
    # line = line.strip().lower()
    line = [l.strip().lower().split(b'\t') for l in lines]
    source_text, target_text = zip(*line)
    print(source_text)

    source_text = [line.decode('utf-8') for line in source_text]
    target_text = [line.decode('utf-8') for line in target_text]
    print(source_text)

    source_ids = source_tokenizer.texts_to_sequences(source_text)
    target_ids = target_tokenizer.texts_to_sequences(target_text)
    print(source_ids)

    source_ids = tf.keras.preprocessing.sequence.pad_sequences(source_ids, padding='post')
    target_ids = tf.keras.preprocessing.sequence.pad_sequences(target_ids, padding='post')
    print(source_ids)
    return source_ids, target_ids


line_dataset = tf.data.TextLineDataset([path])
line_dataset = line_dataset.shuffle(1000).batch(3)
iter = line_dataset.make_one_shot_iterator()
with tf.Session() as sess:
    lines = iter.get_next()
    lines_res = sess.run(lines)
    print(lines_res)
    print(len(lines_res))
    source_text, target_text = tf.py_func(line_process3, [lines], [tf.int32, tf.int32])
    # source_text = tf.py_func(line_process3, [lines], [tf.string, tf.string])

    source_text = sess.run(source_text)
    # print(lines)
    print(source_text)
    # print(line_split_valus)
