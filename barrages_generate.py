# -*- coding: utf-8 -*-
"""
@date: 2020.1.10
@author: liluoqin
@function:
    generate barrages
"""
import os

import tensorflow as tf

from data_load import get_batch, input_fn
from model import Transformer
from hparams import Hparams
from utils import get_hypotheses, calc_bleu, postprocess, load_hparams
import logging

logging.basicConfig(level=logging.INFO)

logging.info("# hparams")
hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()
load_hparams(hp, hp.ckpt)


def gen_barrage(raw_input):
    bar_input = []
    bar_input.append(raw_input)

    test_batches = input_fn(bar_input, bar_input, hp.barrages_vocab, 1, shuffle=False)

    iter = tf.data.Iterator.from_structure(test_batches.output_types, test_batches.output_shapes)
    xs, ys = iter.get_next()

    test_init_op = iter.make_initializer(test_batches)

    logging.info("# Load model")
    m = Transformer(hp)
    y_hat, _ = m.eval(xs, ys)

    logging.info("# Session")
    with tf.Session() as sess:
        ckpt_ = tf.train.latest_checkpoint(hp.ckpt)
        ckpt = hp.ckpt if ckpt_ is None else ckpt_ # None: ckpt is a file. otherwise dir.
        saver = tf.train.Saver()

        saver.restore(sess, ckpt)

        sess.run(test_init_op)

        logging.info("# get hypotheses")
        hypotheses = get_hypotheses(1, 1, sess, y_hat, m.idx2token)

        logging.info("# write results")
        logging.info(hypotheses)


if __name__ == "__main__":
    logging.info("# Prepare input sentence")
    raw_barrages = input()
    gen_barrage(raw_barrages)
