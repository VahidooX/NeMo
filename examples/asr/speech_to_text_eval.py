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

"""
# Preparing the Tokenizer for the dataset
Use the `process_asr_text_tokenizer.py` script under <NEMO_ROOT>/scripts in order to prepare the tokenizer.

```sh

# ["/home/smajumdar/PycharmProjects/nemo-eval/nemo_eval/librispeech/manifests/dev_clean.json","/home/smajumdar/PycharmProjects/nemo-eval/nemo_eval/librispeech/manifests/dev_other.json","/home/smajumdar/PycharmProjects/nemo-eval/nemo_eval/librispeech/manifests/test_clean.json","/home/smajumdar/PycharmProjects/nemo-eval/nemo_eval/librispeech/manifests/test_other.json"]

# Evaluating the model
```sh

python speech_to_text_bpe_eval.py \
    --config-path="experimental/configs/contextnet_bpe/" \
    --config-name="contextnet_192_8x_stride" \
    model.train_ds.manifest_filepath=null \
    model.validation_ds.manifest_filepath=null \
    model.test_ds.manifest_filepath=["/home/smajumdar/PycharmProjects/nemo-eval/nemo_eval/librispeech/manifests/dev_other.json"] \
    model.test_ds.batch_size=1 \
    +model.test_ds.num_workers=12 \
    +model.test_ds.pin_memory=true \
    model.tokenizer.dir=null \
    model.tokenizer.type=wpe \
    +resume_from_checkpoint=null \
    +resume_from_nemo="/home/smajumdar/PycharmProjects/nemo-eval/nemo_beta_eval/librispeech/pretrained/CitriNet-WPE-CTC/Citrinet-384-WPE-1024-8x-Stride_LibriSpeech_Same_Filters.nemo" \
    trainer.gpus=1 \
    trainer.accelerator=null

```
"""
import pytorch_lightning as pl
from omegaconf import OmegaConf, open_dict
import torch

from nemo.collections.asr.models import EncDecCTCModel, EncDecCTCModelBPE
from nemo.core.config import hydra_runner
from nemo.utils import logging


@hydra_runner(config_path="experimental/configs/", config_name="config_bpe")
def main(cfg):

    trainer = pl.Trainer(**cfg.trainer)

    with open_dict(cfg):
        path = cfg.pop('resume_from_checkpoint', None)

        if path is not None:
            ckpt = torch.load(path)

            asr_model = EncDecCTCModelBPE(cfg=cfg.model, trainer=trainer)

            asr_model.load_state_dict(ckpt['state_dict'], strict=False)
            asr_model.freeze()
            del ckpt

        nemo_path = cfg.pop('resume_from_nemo', None)

        if nemo_path is not None:
            asr_model = EncDecCTCModelBPE.restore_from(nemo_path, map_location=torch.device('cpu'))
            asr_model.set_trainer(trainer)
            asr_model.setup_multiple_test_data(cfg.model.test_ds)

        asr_model.preprocessor.featurizer.dither = 0.0
        asr_model.preprocessor.featurizer.pad_to = 0

        if path is None and nemo_path is None:
            raise RuntimeError("No paths provided ! Provide `resume_from_checkpoint` or `resume_from_nemo` !")

    if asr_model.prepare_test(trainer):
        trainer.test(asr_model)


if __name__ == '__main__':
    main()
