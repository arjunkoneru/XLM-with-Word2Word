# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Translate sentences from the input stream.
# The model will be faster is sentences are sorted by length.
# Input sentences must have the same tokenization and BPE codes than the ones used in the model.
#
# Usage:
#     cat source_sentences.bpe | \
#     python translate.py --exp_name translate \
#     --src_lang en --tgt_lang fr \
#     --model_path trained_model.pth --output_path output
#

import os
import io
import sys
import argparse
import torch

from xlm.utils import AttrDict
from xlm.utils import bool_flag, initialize_exp
from xlm.data.dictionary import Dictionary
from xlm.model.transformer import TransformerModel


def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Evaluate sentences with RTTL")

    # main parameters
    parser.add_argument("--dump_path", type=str, default="./dumped/", help="Experiment dump path")
    parser.add_argument("--exp_name", type=str, default="", help="Experiment name")
    parser.add_argument("--exp_id", type=str, default="", help="Experiment ID")
    parser.add_argument("--batch_size", type=int, default=32, help="Number of sentences per batch")

    # model / output paths
    parser.add_argument("--model_path", type=str, default="", help="Model path")
    parser.add_argument("--output_path", type=str, default="", help="Output path for scores")
    parser.add_argument("--input_path", type=str, default="", help="Input path for source sentences")

    # parser.add_argument("--max_vocab", type=int, default=-1, help="Maximum vocabulary size (-1 to disable)")
    # parser.add_argument("--min_count", type=int, default=0, help="Minimum vocabulary count")

    # source language / target language
    parser.add_argument("--src_lang", type=str, default="", help="Source language")
    parser.add_argument("--tgt_lang", type=str, default="", help="Target language")

    return parser


def main(params):

    # initialize the experiment
    logger = initialize_exp(params)

    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()
    reloaded = torch.load(params.model_path)
    model_params = AttrDict(reloaded['params'])
    logger.info("Supported languages: %s" % ", ".join(model_params.lang2id.keys()))

    # update dictionary parameters
    for name in ['n_words', 'bos_index', 'eos_index', 'pad_index', 'unk_index', 'mask_index']:
        setattr(params, name, getattr(model_params, name))

    # build dictionary / build encoder / build decoder / reload weights
    dico = Dictionary(reloaded['dico_id2word'], reloaded['dico_word2id'], reloaded['dico_counts'])
    encoder = TransformerModel(model_params, dico, is_encoder=True, with_output=True).cuda().eval()
    decoder = TransformerModel(model_params, dico, is_encoder=False, with_output=True).cuda().eval()
    encoder.load_state_dict(reloaded['encoder'])
    decoder.load_state_dict(reloaded['decoder'])
    params.src_id = model_params.lang2id[params.src_lang]
    params.tgt_id = model_params.lang2id[params.tgt_lang]
    lens = []
    sent_logprobs = []
    # read sentences from stdin

    with open(params.input_path,'r',encoding = 'utf-8') as f:
            src_sent = f.readlines()
    for x in src_sent:
        assert len(x.strip().split()) > 0

    logger.info("Read %i sentences from file. Scoring ..." % len(src_sent))

    f = io.open(params.output_path, 'w', encoding='utf-8')

    for i in range(0, len(src_sent), params.batch_size):

        # prepare batch
        word_ids = [torch.LongTensor([dico.index(w) for w in s.strip().split()])
                    for s in src_sent[i:i + params.batch_size]]
        lengths = torch.LongTensor([len(s) + 2 for s in word_ids])
        batch = torch.LongTensor(lengths.max().item(), lengths.size(0)).fill_(params.pad_index)
        batch[0] = params.eos_index
        for j, s in enumerate(word_ids):
            if lengths[j] > 2:  # if sentence not empty
                batch[1:lengths[j] - 1, j].copy_(s)
            batch[lengths[j] - 1, j] = params.eos_index
        langs = batch.clone().fill_(params.src_id)

        # encode source batch and translate it
        encoded = encoder('fwd', x=batch.cuda(), lengths=lengths.cuda(), langs=langs.cuda(), causal=False)
        encoded = encoded.transpose(0, 1)
        decoded, dec_lengths = decoder.generate(encoded, lengths.cuda(), params.tgt_id, max_len=int(1.5 * lengths.max().item() + 10))

        langs2 = decoded.clone().fill_(params.tgt_id)
        del encoded
        encoded2 = encoder('fwd', x=decoded.cuda(), lengths = dec_lengths.cuda(), langs=langs2.cuda(), causal=False)
        encoded2 = encoded2.transpose(0,1)
        alen = torch.arange(lengths.max(), dtype=torch.long, device=lengths.device)
        pred_mask = alen[:, None] < lengths[None] - 1
        y1 = batch[1:].masked_select(pred_mask[:-1])
        dec2 = decoder('fwd', x=batch.cuda(), lengths=lengths.cuda(), langs=langs.cuda(), causal=True, src_enc=encoded2, src_len=dec_lengths.cuda())
        scores, loss = decoder('predict', tensor=dec2.cuda(), pred_mask=pred_mask.cuda(), y=y1.cuda(), get_scores=True)
        logprobs = torch.gather(scores, 1, y1.unsqueeze(-1).cuda() ,out=None, sparse_grad=False)
        j = 0
        for sent in lengths:
            idx = sent.item()
            sent_logprob = torch.div(torch.sum(logprobs[j:j+idx-1]),idx-1)
            sent_logprobs.append(sent_logprob)
            j+= idx
            lens.append(idx - 1)
    f.writelines(sent_logprobs)
    f.close()



if __name__ == '__main__':

    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()

    # check parameters
    assert os.path.isfile(params.model_path)
    assert params.src_lang != '' and params.tgt_lang != '' and params.src_lang != params.tgt_lang
    #assert params.output_path and not os.path.isfile(params.output_path)

    # translate
    with torch.no_grad():
        main(params)
