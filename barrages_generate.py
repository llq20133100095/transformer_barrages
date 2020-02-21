# -*- coding: utf-8 -*-
"""
@date: 2020.1.10
@author: liluoqin
@function:
    generate pretreatment
"""
import os

import tensorflow as tf

from data_load import get_batch, input_fn
from model import Transformer
from hparams import Hparams
from utils import get_hypotheses, load_hparams, pro_sentpiece
import logging
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

logging.basicConfig(level=logging.INFO)

logging.info("# hparams")
hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()
load_hparams(hp, hp.ckpt)


def gen_barrage(raw_input):
    raw_input = pro_sentpiece(raw_input, hp.bpe_model)
    bar_input = []
    bar_input.append(raw_input)

    test_batches = input_fn(bar_input, bar_input, hp.barrages_vocab, 1, shuffle=False)

    iter = tf.data.Iterator.from_structure(test_batches.output_types, test_batches.output_shapes)
    xs, ys = iter.get_next()

    test_init_op = iter.make_initializer(test_batches)

    logging.info("# Load model")
    m = Transformer(hp)
    y_hat, _, random_predict = m.eval_gen(xs, ys)

    logging.info("# Session")

    with tf.Session() as sess:
        ckpt_ = tf.train.latest_checkpoint(hp.ckpt)
        ckpt = hp.ckpt if ckpt_ is None else ckpt_ # None: ckpt is a file. otherwise dir.
        saver = tf.train.Saver()

        saver.restore(sess, ckpt)

        sess.run(test_init_op)
        y_output = sess.run(random_predict)

        logging.info("# get hypotheses")
        hypotheses, yy = get_hypotheses(1, 1, sess, y_hat, m.idx2token)

        logging.info("# write results")
        logging.info(hypotheses)

        logging.info("# Done")

    return "".join(hypotheses)


def init_sess_model():
    logging.info("# Load Transformer")
    m = Transformer(hp)

    return m


def gen_barrage_wechcat(raw_input, model):
    raw_input = pro_sentpiece(raw_input, hp.bpe_model)
    bar_input = []
    bar_input.append(raw_input)

    test_batches = input_fn(bar_input, bar_input, hp.barrages_vocab, 1, shuffle=False)

    iter = tf.data.Iterator.from_structure(test_batches.output_types, test_batches.output_shapes)
    xs, ys = iter.get_next()

    test_init_op = iter.make_initializer(test_batches)
    y_hat, _, random_predict = model.eval_gen(xs, ys)

    with tf.Session() as sess:
        ckpt_ = tf.train.latest_checkpoint(hp.ckpt)
        ckpt = hp.ckpt if ckpt_ is None else ckpt_  # None: ckpt is a file. otherwise dir.
        saver = tf.train.Saver()
        saver.restore(sess, ckpt)

        sess.run(test_init_op)

        logging.info("# get hypotheses")
        hypotheses, yy = get_hypotheses(1, 1, sess, y_hat, model.idx2token)

        logging.info("# write results")
        logging.info(hypotheses)

        logging.info("# Done")

    return "".join(hypotheses)


if __name__ == "__main__":
    logging.info("# Prepare input sentence")
    raw_barrages = "老司机"
    # raw_barrages = pro_sentpiece(raw_barrages, hp.bpe_model)
    output = gen_barrage(raw_barrages)

    # # use in wechat
    # transformer = init_sess_model()
    # output = gen_barrage_wechcat(raw_barrages, transformer)