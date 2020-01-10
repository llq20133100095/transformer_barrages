# -*- coding: utf-8 -*-
"""
@date: 2020.1.10
@author: liluoqin
@function:
    process barrages data
"""
import os
import errno
import sentencepiece as spm
import re
from hparams import Hparams
import logging
import jieba

logging.basicConfig(level=logging.INFO)

def prepro(hp):

    barrages_data = hp.barrages_data

    # train
    _prepro = lambda x: [line.split("\t")[0] for line in open(x, 'r').readline().strip()
                         if not line.startswith("barrage")]

    def _write(sents, fname):
        with open(fname, 'w') as fout:
            fout.write("\n".join(sents))

    logging.info("# Preprocessing")
    prepro_sents = _prepro(barrages_data)

    logging.info("# write preprocessed files to disk")
    os.makedirs("../barrages_data/prepro", exist_ok=True)

    _write(prepro_sents, "../barrages_data/prepro/train.txt")
    _write(prepro_sents, "../barrages_data/prepro/test.de")

    # logging.info("# Train a joint BPE model with sentencepiece")
    # os.makedirs("iwslt2016/segmented", exist_ok=True)
    # train = '--input=../barrages_data/prepro/train --pad_id=0 --unk_id=1 \
    #              --bos_id=2 --eos_id=3\
    #              --model_prefix=iwslt2016/segmented/bpe --vocab_size={} \
    #              --model_type=bpe'.format(hp.vocab_size)
    # spm.SentencePieceTrainer.Train(train)
    #
    # logging.info("# Load trained bpe model")
    # sp = spm.SentencePieceProcessor()
    # sp.Load("iwslt2016/segmented/bpe.model")

    # logging.info("# Segment")


if __name__ == "__main__":
    hparams = Hparams()
    parser = hparams.parser
    hp = parser.parse_args()
    prepro(hp)
    logging.info("# Done")