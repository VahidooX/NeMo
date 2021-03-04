# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import math
import pandas as pd
import numpy as np
from typing import Dict, Optional

import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer

from nemo.collections.common.losses import SmoothedCrossEntropyLoss
from nemo.collections.common.metrics import Perplexity
from nemo.collections.common.parts import transformer_weights_init
from nemo.collections.nlp.data import L2RLanguageModelingDataset, TarredL2RLanguageModelingDataset
from nemo.collections.nlp.modules.common import TokenClassifier
from nemo.collections.nlp.modules.common.tokenizer_utils import get_tokenizer
from nemo.collections.nlp.modules.common.transformer import TransformerEmbedding, TransformerEncoder
from nemo.core.classes.common import typecheck
from nemo.core.classes.modelPT import ModelPT
from nemo.utils import logging

__all__ = ["TransformerLMModel"]


class BeamSearchDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, tokenizer, max_seq_length=256):
        self.data = pd.read_csv(data_path, delimiter="\t", header=None)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = [self.tokenizer.bos_id] + \
                 self.tokenizer.text_to_ids(str(self.data[0][idx])) + [self.tokenizer.eos_id]
        input_ids = [self.tokenizer.pad_id] * self.max_seq_length
        input_ids[: len(tokens)] = tokens
        input_ids = np.array(input_ids)
        input_mask = (input_ids != self.tokenizer.pad_id).astype(np.float32)
        score = self.data[1][idx]
        dist = self.data[2][idx]
        ref_len = self.data[3][idx]
        len_in_chars = len(str(self.data[0][idx]))
        return input_ids, input_mask, score, dist, ref_len, len_in_chars, idx


class TransformerLMModel(ModelPT):
    """
    Left-to-right Transformer language model.
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):

        # Get global rank and total number of GPU workers for IterableDataset partitioning, if applicable
        self.global_rank = 0
        self.world_size = 1
        if trainer is not None:
            self.global_rank = (trainer.node_rank * trainer.num_gpus) + trainer.local_rank
            self.world_size = trainer.num_nodes * trainer.num_gpus

        # shared params for dataset and data loaders
        self.dataset_cfg = cfg.dataset
        self.tokenizer = get_tokenizer(
            tokenizer_name=cfg.language_model.tokenizer,
            vocab_file=cfg.language_model.get("vocab_file", None),
            tokenizer_model=cfg.language_model.get("tokenizer_model", None),
            special_tokens=cfg.language_model.special_tokens,
        )

        # make vocabulary size divisible by 8 for fast fp16 training
        vocab_size = 8 * math.ceil(self.tokenizer.vocab_size / 8)

        # init superclass
        super().__init__(cfg=cfg, trainer=trainer)

        self.embedding_layer = TransformerEmbedding(
            vocab_size=vocab_size,
            hidden_size=cfg.language_model.hidden_size,
            max_sequence_length=cfg.language_model.max_seq_length,
            embedding_dropout=cfg.language_model.get("embedding_dropout", 0.0),
            learn_positional_encodings=False,
        )
        self.encoder = TransformerEncoder(
            num_layers=cfg.language_model.num_layers,
            hidden_size=cfg.language_model.hidden_size,
            mask_future=True,
            num_attention_heads=cfg.language_model.num_attn_heads,
            inner_size=cfg.language_model.inner_size,
            ffn_dropout=cfg.language_model.get("ffn_dropout", 0.0),
            hidden_act=cfg.language_model.get("inner_activation", "relu"),
            attn_score_dropout=cfg.language_model.get("attn_score_dropout", 0.0),
            attn_layer_dropout=cfg.language_model.get("attn_layer_dropout", 0.0),
        )
        self.log_softmax = TokenClassifier(
            hidden_size=cfg.language_model.hidden_size, num_classes=vocab_size, log_softmax=True,
        )

        std_init_range = 1 / math.sqrt(cfg.language_model.hidden_size)
        self.apply(lambda module: transformer_weights_init(module, std_init_range))

        # tie weights of embedding and softmax matrices
        self.log_softmax.mlp.layer0.weight = self.embedding_layer.token_embedding.weight

        self.training_loss = SmoothedCrossEntropyLoss(pad_id=self.tokenizer.pad_id)
        self.validation_loss = SmoothedCrossEntropyLoss(
            pad_id=self.tokenizer.pad_id, predict_last_k=self.dataset_cfg.get("predict_last_k", 0),
        )

        self.training_perplexity = Perplexity(dist_sync_on_step=True)
        self.validation_perplexity = Perplexity(compute_on_step=False)

        # Optimizer setup needs to happen after all model weights are ready
        self.setup_optimization()

    @typecheck()
    def forward(self, input_ids, attention_mask):
        """
        No special modification required for Lightning, define it as you normally would
        in the `nn.Module` in vanilla PyTorch.
        """
        token_embeddings = self.embedding_layer(input_ids)
        hidden_states = self.encoder(token_embeddings, attention_mask)
        log_probs = self.log_softmax(hidden_states=hidden_states)

        return log_probs

    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop with the data from the training dataloader
        passed in as `batch`.
        """
        # forward pass
        input_ids, input_mask, labels = batch
        log_probs = self(input_ids=input_ids, attention_mask=input_mask)

        target_log_probs = log_probs.gather(2, input_ids[:, 1:].unsqueeze(2)).squeeze(2)
        lm_scores = torch.sum(target_log_probs * input_mask[:, :-1], dim=-1)

        train_loss = self.training_loss(log_probs=log_probs, labels=labels)
        training_perplexity = self.training_perplexity(logits=log_probs)

        tensorboard_logs = {
            "train_loss": train_loss,
            "lr": self._optimizer.param_groups[0]["lr"],
            "train_ppl": training_perplexity,
        }
        return {"loss": train_loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        input_ids, input_mask, scores, dist, ref_len, len_in_chars, idxs = batch

        log_probs = self(input_ids=input_ids[:, :-1], attention_mask=input_mask[:, :-1])
        val_loss = self.validation_loss(log_probs=log_probs, labels=input_ids[:, 1:])
        self.validation_perplexity(logits=log_probs)

        target_log_probs = log_probs.gather(2, input_ids[:, 1:].unsqueeze(2)).squeeze(2)
        lm_scores = torch.sum(target_log_probs * input_mask[:, :-1], dim=-1)

        tensorboard_logs = {
            "val_loss": val_loss,
        }

        return {"val_loss": val_loss,
                "am_scores": scores,
                "lm_scores": lm_scores,
                "dist": dist,
                "ref_len": ref_len,
                "len_in_chars": len_in_chars,
                "idxs": idxs,
                "log": tensorboard_logs}

    def validation_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs.
        :param outputs: list of individual outputs of each validation step.
        """

        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        validation_perplexity = self.validation_perplexity.compute()

        idxs = torch.cat([x["idxs"] for x in outputs])
        dist = torch.cat([x["dist"] for x in outputs])
        ref_len = torch.cat([x["ref_len"] for x in outputs])
        len_in_chars = torch.cat([x["len_in_chars"] for x in outputs])
        ints = torch.stack([idxs, dist, ref_len, len_in_chars])

        am_scores = torch.cat([x["am_scores"] for x in outputs])
        lm_scores = torch.cat([x["lm_scores"] for x in outputs])
        scores = torch.stack([am_scores, lm_scores])

        all_ints, all_scores = [], []
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            for ind in range(world_size):
                all_ints.append(torch.empty_like(ints))
                all_scores.append(torch.empty_like(scores))
            torch.distributed.all_gather(all_ints, ints)
            torch.distributed.all_gather(all_scores, scores)
        else:
            all_ints.append(ints)
            all_scores.append(scores)

        model_wer, ideal_wer, worst_wer, lm_wer, coef1, coef2 = 0, 0, 0, 0, 0, 0

        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            idxs, dist, ref_len, len_in_chars = torch.cat(all_ints, dim=1)
            am_scores, lm_scores = torch.cat(all_scores, dim=1)

            idxs = idxs.sort()[1]
            dist, ref_len, len_in_chars = dist[idxs], ref_len[idxs], len_in_chars[idxs]
            am_scores, lm_scores = am_scores[idxs], lm_scores[idxs]

            am_scores = am_scores.view(-1, self.dataset_cfg.beam_size)
            lm_scores = lm_scores.view(-1, self.dataset_cfg.beam_size)
            dist = dist.view(-1, self.dataset_cfg.beam_size).to(am_scores.dtype)
            ref_len = ref_len.view(-1, self.dataset_cfg.beam_size).to(am_scores.dtype)
            len_in_chars = len_in_chars.view(-1, self.dataset_cfg.beam_size).to(am_scores.dtype)
            total_len = ref_len[:, 0].sum()

            model_wer = dist[:, 0].sum() / total_len
            ideal_wer = dist.min(dim=1)[0].sum() / total_len
            worst_wer = dist.max(dim=1)[0].sum() / total_len

            coef1, wer1 = self.line_search_wer(dist, am_scores, lm_scores, total_len)
            scores = am_scores + coef1 * lm_scores
            coef2, lm_wer = self.line_search_wer(dist, scores, len_in_chars, total_len)

            model_wer, ideal_wer, worst_wer = model_wer.item(), ideal_wer.item(), worst_wer.item()

            logging.info("\n\n\n\n")
            logging.info(f"     AM+n_gram WER: {np.round(model_wer * 100, 2)}")
            logging.info(f" +LM rescoring WER: {np.round(lm_wer * 100, 2)}")
            logging.info(f" Best possible WER: {np.round(ideal_wer * 100, 2)}")

        tensorboard_logs = {
            "val_loss": avg_loss,
            "val_ppl": validation_perplexity,
            "model_wer": model_wer,
            "ideal_wer": ideal_wer,
            "worst_wer": worst_wer,
            "lm_wer": lm_wer,
            "neural_lm_coef": coef1,
            "len_in_chars_coef": coef2,
        }

        return {"val_loss": avg_loss, "lm_wer": lm_wer, "log": tensorboard_logs}

    def line_search_wer(self, dist, scores1, scores2, total_len=1):

        scale = scores1.mean().abs().item() / scores2.mean().abs().item()
        left = self.dataset_cfg.coef_range[0] * scale
        right = self.dataset_cfg.coef_range[1] * scale
        coefs = np.linspace(left, right, self.dataset_cfg.coef_steps)

        best_wer = 10000
        best_coef = left
        for coef in coefs:
            scores = scores1 + coef * scores2
            indices = scores.max(dim=1, keepdim=True)[1]
            wer = dist.gather(dim=1, index=indices).sum() / total_len
            wer = wer.item()
            if wer < best_wer:
                best_wer = wer
                best_coef = coef
        return best_coef, best_wer

    def test_step(self, batch, batch_idx):
        """
        Lightning calls this inside the test loop with the data from the test dataloader
        passed in as `batch`.
        """
        output_dict = self.validation_step(batch, batch_idx)
        result = {"test_loss": output_dict['val_loss'], "log": {}}
        for k, v in output_dict['log'].items():
            new_k = k.replace("val", "test")
            result['log'][new_k] = v

        return result

    def test_epoch_end(self, outputs):
        """
        Called at the end of test step to aggregate outputs.
        :param outputs: list of individual outputs of each test step.
        """

        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        validation_perplexity = self.validation_perplexity.compute()
        tensorboard_logs = {"test_loss": avg_loss, "test_ppl": validation_perplexity}
        return {"test_loss": avg_loss, "log": tensorboard_logs}

    def setup_training_data(self, train_data_config: Optional[DictConfig]):
        self._train_dl = self._setup_dataloader_from_config(cfg=train_data_config)

        # Need to set this because if using an IterableDataset, the length of the dataloader is the total number
        # of samples rather than the number of batches, and this messes up the tqdm progress bar.
        # So we set the number of steps manually (to the correct number) to fix this.
        if 'is_tarred' in train_data_config and train_data_config['is_tarred']:
            # We also need to check if limit_train_batches is already set.
            # If it's an int, we assume that the user has set it to something sane, i.e. <= # training batches,
            # and don't change it. Otherwise, adjust batches accordingly if it's a float (including 1.0).
            if isinstance(self._trainer.limit_train_batches, float):
                self._trainer.limit_train_batches = int(
                    self._trainer.limit_train_batches
                    * math.ceil((len(self._train_dl.dataset) / self.world_size) / train_data_config['batch_size'])
                )

    def setup_validation_data(self, val_data_config: Optional[DictConfig]):
        self._validation_dl = self._setup_val_dataloader(
            cfg=val_data_config, predict_last_k=self.dataset_cfg.get("predict_last_k", 0),
        )

    def setup_test_data(self, test_data_config: Optional[DictConfig]):
        self._test_dl = self._setup_dataloader_from_config(
            cfg=test_data_config, predict_last_k=self.dataset_cfg.get("predict_last_k", 0),
        )

    def _setup_val_dataloader(self, cfg: DictConfig, predict_last_k=0):
        dataset = BeamSearchDataset(
            data_path=cfg.file_name,
            tokenizer=self.tokenizer,
            max_seq_length=self.dataset_cfg.max_seq_length
        )
        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=self.dataset_cfg.get("num_workers", 2),
            pin_memory=self.dataset_cfg.get("pin_memory", False),
            drop_last=self.dataset_cfg.get("drop_last", False),
        )

    def _setup_dataloader_from_config(self, cfg: DictConfig, predict_last_k=0):
        if cfg.get('use_cache', False):
            logging.info("Constructing tokenized dataset cache...")

        shuffle = cfg.shuffle

        if cfg.get('is_tarred', False):
            if ('tarred_text_filepaths' in cfg and cfg['tarred_text_filepaths'] is None) or (
                    'file_name' in cfg and cfg['file_name'] is None
            ):
                logging.warning(
                    "Could not load dataset as `file_name` was None or "
                    f"`tarred_text_filepaths` is None. Provided config : {cfg}"
                )
                return None

            shuffle_n = cfg.get('shuffle_n', 4 * cfg['batch_size']) if shuffle else 0
            dataset = TarredL2RLanguageModelingDataset(
                text_tar_filepaths=cfg['tarred_text_filepaths'],
                metadata_path=cfg['file_name'],
                tokenizer=self.tokenizer,
                max_seq_length=self.dataset_cfg.max_seq_length,
                batch_step=predict_last_k,
                shuffle_n=shuffle_n,
                shard_strategy=cfg.get("tarred_shard_strategy", "scatter"),
                global_rank=self.global_rank,
                world_size=self.world_size,
            )

            shuffle = False
        else:

            dataset = L2RLanguageModelingDataset(
                tokenizer=self.tokenizer,
                dataset=cfg.file_name,
                max_seq_length=self.dataset_cfg.max_seq_length,
                batch_step=predict_last_k,
                use_cache=cfg.get('use_cache', False),
            )

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=cfg.batch_size,
            shuffle=shuffle,
            num_workers=self.dataset_cfg.get("num_workers", 2),
            pin_memory=self.dataset_cfg.get("pin_memory", False),
            drop_last=self.dataset_cfg.get("drop_last", False),
        )

    @classmethod
    def list_available_models(cls) -> Optional[Dict[str, str]]:
        pass
