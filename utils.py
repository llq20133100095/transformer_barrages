# -*- coding: utf-8 -*-
# /usr/bin/python3
'''
Feb. 2019 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer.

Utility functions
'''

import tensorflow as tf
# from tensorflow.python import pywrap_tensorflow
# import numpy as np
import json
import os, re
import logging
import sentencepiece as spm
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import corpus_bleu
logging.basicConfig(level=logging.INFO)

def calc_num_batches(total_num, batch_size):
    '''Calculates the number of batches.
    total_num: total sample number
    batch_size

    Returns
    number of batches, allowing for remainders.'''
    return total_num // batch_size + int(total_num % batch_size != 0)

def convert_idx_to_token_tensor(inputs, idx2token):
    '''Converts int32 tensor to string tensor.
    inputs: 1d int32 tensor. indices.
    idx2token: dictionary

    Returns
    1d string tensor.
    '''
    def my_func(inputs):
        return " ".join(idx2token[elem] for elem in inputs)

    return tf.py_func(my_func, [inputs], tf.string)

# # def pad(x, maxlen):
# #     '''Pads x, list of sequences, and make it as a numpy array.
# #     x: list of sequences. e.g., [[2, 3, 4], [5, 6, 7, 8, 9], ...]
# #     maxlen: scalar
# #
# #     Returns
# #     numpy int32 array of (len(x), maxlen)
# #     '''
# #     padded = []
# #     for seq in x:
# #         seq += [0] * (maxlen - len(seq))
# #         padded.append(seq)
# #
# #     arry = np.array(padded, np.int32)
# #     assert arry.shape == (len(x), maxlen), "Failed to make an array"
#
#     return arry

def postprocess(hypotheses, idx2token):
    '''Processes translation outputs.
    hypotheses: list of encoded predictions
    idx2token: dictionary

    Returns
    processed hypotheses
    '''
    _hypotheses = []
    for h in hypotheses:
        sent = " ".join(idx2token[idx] for idx in h)
        sent = sent.split("</s>")[0].strip()
        sent = sent.replace("▁", "") # remove bpe symbols
        _hypotheses.append(sent.strip())
    return _hypotheses

def save_hparams(hparams, path):
    '''Saves hparams to path
    hparams: argsparse object.
    path: output directory.

    Writes
    hparams as literal dictionary to path.
    '''
    if not os.path.exists(path): os.makedirs(path)
    hp = json.dumps(vars(hparams))
    with open(os.path.join(path, "hparams"), 'w') as fout:
        fout.write(hp)

def load_hparams(parser, path):
    '''Loads hparams and overrides parser
    parser: argsparse parser
    path: directory or file where hparams are saved
    '''
    if not os.path.isdir(path):
        path = os.path.dirname(path)
    d = open(os.path.join(path, "hparams"), 'r').read()
    flag2val = json.loads(d)
    for f, v in flag2val.items():
        parser.f = v

def save_variable_specs(fpath):
    '''Saves information about variables such as
    their name, shape, and total parameter number
    fpath: string. output file path

    Writes
    a text file named fpath.
    '''
    def _get_size(shp):
        '''Gets size of tensor shape
        shp: TensorShape

        Returns
        size
        '''
        size = 1
        for d in range(len(shp)):
            size *=shp[d]
        return size

    params, num_params = [], 0
    for v in tf.global_variables():
        params.append("{}==={}".format(v.name, v.shape))
        num_params += _get_size(v.shape)
    print("num_params: ", num_params)
    with open(fpath, 'w') as fout:
        fout.write("num_params: {}\n".format(num_params))
        fout.write("\n".join(params))
    logging.info("Variables info has been saved.")

def get_hypotheses(num_batches, num_samples, sess, tensor, dict):
    '''Gets hypotheses.
    num_batches: scalar.
    num_samples: scalar.
    sess: tensorflow sess object
    tensor: target tensor to fetch
    dict: idx2token dictionary

    Returns
    hypotheses: list of sents
    '''
    hypotheses = []
    for _ in range(num_batches):
        h = sess.run(tensor)
        hypotheses.extend(h.tolist())
    hypotheses = postprocess(hypotheses, dict)

    return hypotheses[:num_samples], h

def calc_bleu(ref, translation):
    '''Calculates bleu score and appends the report to translation
    ref: reference file path
    translation: model output file path

    Returns
    translation that the bleu score is appended to'''
    get_bleu_score = "perl multi-bleu.perl {} < {} > {}".format(ref, translation, "temp")
    os.system(get_bleu_score)
    bleu_score_report = open("temp", "r").read()
    with open(translation, "a") as fout:
        fout.write("\n{}".format(bleu_score_report))
    try:
        score = re.findall("BLEU = ([^,]+)", bleu_score_report)[0]
        new_translation = translation + "B{}".format(score)
        os.system("mv {} {}".format(translation, new_translation))
        os.remove(translation)

    except: pass
    os.remove("temp")


def random_id(predict):
    """
    randomly get a sample
    :param predict: (N, T, vocab_size)
    :return:
        sample: (N, T)
    """
    predict_exp = tf.exp(predict)
    N, T, vocab_size = tf.shape(predict)[0], tf.shape(predict)[1], tf.shape(predict)[2]
    pre_mol= tf.transpose(predict_exp, perm=[0, 2, 1])  # (N, vocab_size, T)
    pre_den = tf.reshape(tf.reduce_sum(predict_exp, axis=-1), [N, 1, T])    # (N, 1, T)
    predict_exp = tf.reshape(tf.transpose(tf.div(pre_mol, pre_den), perm=[0, 2, 1]), [-1, vocab_size]) # (N * T, vocab_size)
    sample = tf.to_int32(tf.reshape(tf.multinomial(predict_exp, 1), [N, T]))
    return sample


def pro_sentpiece(sent, bep_model):
    """
    Process input sentence. Use the sentencepiece to segment sentence.
    :param sent:
    :param bep_model:
    :return: string, has pieces
    """
    sp = spm.SentencePieceProcessor()
    sp.Load(bep_model)
    pieces = sp.EncodeAsPieces(sent)
    return " ".join(pieces)


def calc_belu_nltk(bep_model, ref, translation):
    """
    calculate BELU-2 score.
    :param bep_model:
    :param ref: list, reference sentences
    :param translation: list, model generate sentences
    :return: score: float
    """
    # read ref file
    ref_sent = []
    with open(ref, "r", encoding="utf-8") as f:
        while True:
            line = f.readline()
            if line:
                line = line.replace("▁", "")
                ref_sent.append(line.strip())
            else:
                break

    ref_pieces_ = [[i.split()] for i in ref_sent]
    trans_pieces_ = [i.split() for i in translation]

    score = corpus_bleu(ref_pieces_, trans_pieces_, weights=(0.5, 0.5, 0, 0))
    logging.info("# BELU-2: %f" % score)
    return score


def plot_fig(x, y, title, save_fig):
    plt.figure(figsize=(10, 10))
    plt.title(title)
    plt.grid()  # open grid
    plt.xlabel('number epoch')
    plt.ylabel("BELU-2")
    plt.plot(x, y)
    plt.savefig(save_fig)