# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script would evaluate a neural language model (Transformer) trained with
`examples/nlp/language_modeling/transformer_lm.py' as a rescorer for ASR systems.
Given a trained TransformerLMModel `.nemo` file, this script can be used to re-score the beams obtained from a beam
search decoder of an ASR model.

USAGE:
1. Obtain `.tsv` file with beams and their corresponding scores. Scores can be from a regular beam search decoder or
   in fusion with an N-gram LM scores. For a given beam size `beam_size` and a number of examples
   for evaluation `num_eval_examples`, it should contain (`beam_size` x `num_eval_examples`) lines of
   form `beam_candidate_text \t score`. This file can be generated by `scripts/asr_language_modeling/ngram_lm/eval_beamsearch_ngram.py`.

2. Rescore the candidates:
    python eval_neural_rescorer.py
        --lm_model=[path to .nemo file of the LM]
        --beams_file=[path to beams .tsv file]
        --beam_size=[size of the beams]
        --eval_manifest=[path to eval manifest .json file]
        --batch_size=[batch size used for inference on the LM model]
        --alpha=[the value for the parameter rescorer_alpha]
        --beta=[the value for the parameter rescorer_beta]

You may find more info on how to use this script at:
https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/asr_language_modeling.html

"""

import contextlib
import json
from argparse import ArgumentParser

import editdistance
import numpy as np
import pandas as pd
import torch
import tqdm

from nemo.collections.nlp.models.language_modeling import TransformerLMModel
from nemo.utils import logging


class BeamSearchDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, tokenizer, manifest_path, beam_size=128, max_seq_length=256):
        self.data = pd.read_csv(data_path, delimiter="\t", header=None)
        self.tokenizer = tokenizer
        self.ground_truths = []
        with open(manifest_path, 'r') as f_orig:
            for line in f_orig:
                item = json.loads(line)
                self.ground_truths.append(item['text'])
        self.beam_size = beam_size
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = str(self.data[0][idx])
        tokens = [self.tokenizer.bos_id] + self.tokenizer.text_to_ids(text) + [self.tokenizer.eos_id]
        input_ids = [self.tokenizer.pad_id] * self.max_seq_length
        input_ids[: len(tokens)] = tokens
        input_ids = np.array(input_ids)
        input_mask = (input_ids != self.tokenizer.pad_id).astype(np.float32)
        acoustic_score = self.data[1][idx]
        dist = editdistance.eval(text.split(), self.ground_truths[idx // self.beam_size].split())
        ref_len = len(self.ground_truths[idx // self.beam_size].split())
        len_in_chars = len(str(self.data[0][idx]))
        return input_ids, input_mask, acoustic_score, dist, ref_len, len_in_chars, idx


def line_search_wer(dists, scores1, scores2, total_len, coef_range=[0, 10], coef_steps=10000):
    scale = scores1.mean().abs().item() / scores2.mean().abs().item()
    left = coef_range[0] * scale
    right = coef_range[1] * scale
    coefs = np.linspace(left, right, coef_steps)

    best_wer = 10000
    best_coef = left
    for coef in coefs:
        scores = scores1 + coef * scores2
        wer = compute_wer(dists, scores, total_len)
        if wer < best_wer:
            best_wer = wer
            best_coef = coef
    return best_coef, best_wer


def compute_wer(dists, scores, total_len):
    indices = scores.max(dim=1, keepdim=True)[1]
    wer = dists.gather(dim=1, index=indices).sum() / total_len
    wer = wer.item()
    return wer


def main():
    parser = ArgumentParser()
    parser.add_argument("--lm_model_file", type=str, required=True, help="path to LM model .nemo file")
    parser.add_argument("--beams_file", type=str, required=True, help="path to beams .tsv file")
    parser.add_argument(
        "--eval_manifest", type=str, required=True, help="path to the evaluation `.json` manifest file"
    )
    parser.add_argument("--beam_size", type=int, required=True, help="number of beams per candidate")
    parser.add_argument("--batch_size", type=int, default=256, help="inference batch size")
    parser.add_argument("--alpha", type=float, default=None, help="parameter alpha of the fusion")
    parser.add_argument("--beta", type=float, default=None, help="parameter beta of the fusion")
    parser.add_argument(
        "--device", default="cuda", type=str, help="The device to load the model onto to calculate the scores"
    )
    parser.add_argument(
        "--use_amp", action="store_true", help="Whether to use AMP if available to calculate the scores"
    )
    args = parser.parse_args()

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        logging.info(f"cuda is not available! switched to cpu.")
        device = "cpu"

    if args.lm_model_file.endswith(".nemo"):
        logging.info("Attempting to initialize from .nemo file")
        model = TransformerLMModel.restore_from(
            restore_path=args.lm_model_file, map_location=torch.device(device)
        ).eval()
    else:
        raise NotImplementedError(f"Only supports .nemo files, but got: {args.model}")

    max_seq_length = model.encoder._embedding.position_embedding.pos_enc.shape[0]
    dataset = BeamSearchDataset(args.beams_file, model.tokenizer, args.eval_manifest, args.beam_size, max_seq_length)
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.batch_size)

    if args.use_amp:
        if torch.cuda.is_available() and hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast'):
            logging.info("AMP is enabled!\n")
            autocast = torch.cuda.amp.autocast
    else:

        @contextlib.contextmanager
        def autocast():
            yield

    logging.info(f"Rescoring with beam_size: {args.beam_size}")
    logging.info("Calculating the scores...")
    with autocast():
        with torch.no_grad():
            am_scores, lm_scores, dists, ref_lens, lens_in_chars = [], [], [], [], []
            for batch in tqdm.tqdm(data_loader):
                input_ids, input_mask, acoustic_score, dist, ref_len, len_in_chars, idx = batch

                max_len_in_batch = input_mask.sum(dim=0).argmin().item()
                input_ids, input_mask = input_ids[:, :max_len_in_batch], input_mask[:, :max_len_in_batch]
                if torch.cuda.is_available():
                    input_ids, input_mask = input_ids.to(device), input_mask.to(device)
                    dist, acoustic_score, len_in_chars = (
                        dist.to(device),
                        acoustic_score.to(device),
                        len_in_chars.to(device),
                    )

                log_probs = model.forward(input_ids[:, :-1], input_mask[:, :-1])
                target_log_probs = log_probs.gather(2, input_ids[:, 1:].unsqueeze(2)).squeeze(2)
                neural_lm_score = torch.sum(target_log_probs * input_mask[:, 1:], dim=-1)

                am_scores.append(acoustic_score)
                lm_scores.append(neural_lm_score)
                dists.append(dist)
                ref_lens.append(ref_len)
                lens_in_chars.append(len_in_chars)

    am_scores = torch.cat(am_scores).view(-1, args.beam_size)
    lm_scores = torch.cat(lm_scores).view(-1, args.beam_size)
    dists = torch.cat(dists).view(-1, args.beam_size)
    ref_lens = torch.cat(ref_lens).view(-1, args.beam_size)
    lens_in_chars = torch.cat(lens_in_chars).view(-1, args.beam_size).to(am_scores.dtype)

    total_len = ref_lens[:, 0].sum()
    model_wer = dists[:, 0].sum() / total_len
    ideal_wer = dists.min(dim=1)[0].sum() / total_len

    if args.alpha is None:
        logging.info("Linear search for alpha...")
        coef1, _ = line_search_wer(dists=dists, scores1=am_scores, scores2=lm_scores, total_len=total_len)
        logging.info(f"alpha={coef1} achieved the best WER.")
        logging.info(f"------------------------------------------------")
    else:
        coef1 = args.alpha
    coef1 = np.round(coef1, 3)

    scores = am_scores + coef1 * lm_scores

    if args.beta is None:
        logging.info("Linear search for beta...")
        coef2, _ = line_search_wer(dists, scores, lens_in_chars, total_len)
        logging.info(f"beta={coef2} achieved the best WER.")
        logging.info(f"------------------------------------------------")
    else:
        coef2 = args.beta
    coef2 = np.round(coef2, 3)

    ab_scores = am_scores + coef1 * lm_scores + coef2 * lens_in_chars
    ab_wer = compute_wer(dists, ab_scores, total_len)

    logging.info(f"------------------------------------------------")
    logging.info(f"Input beams WER: {np.round(model_wer.item() * 100, 2)}%")
    logging.info(f"------------------------------------------------")
    logging.info(f"  +LM rescoring WER: {np.round(ab_wer * 100, 2)}%")
    logging.info(f"  with alpha={coef1}, beta={coef2}")
    logging.info(f"------------------------------------------------")
    logging.info(f"Best possible WER: {np.round(ideal_wer.item() * 100, 2)}%")
    logging.info(f"------------------------------------------------")


if __name__ == '__main__':
    main()
