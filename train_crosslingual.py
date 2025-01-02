#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
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
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import logging
import numpy as np
import math
import os
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional, List, Dict, Any, Mapping
from pathlib import Path
import shutil
from torch.utils.data import DataLoader
import datasets
import json
from cross_lingual_utils.prompter import Prompter
import torch
from datasets import load_from_disk
from cross_lingual_utils.cross_lingual_trainer import CrossLingualTrainer
from cross_lingual_utils.batch_sampling import GatherLogits, DecomposeLanguageMetric
from cross_lingual_utils.cross_datasets import CrossDataset, create_validation_map, train_val_dataset, MyDataCollator
# from seaeval.seaeval_utils import construct_seaeval_data, SeaEvalTrainer, ComputeSeaevalMetric, WandbPredictionCallback, get_pred_ids
# >>>>>>> seaeval_integration

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    is_torch_tpu_available,
    set_seed,
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import send_example_telemetry
from transformers.utils.versions import require_version

from sklearn.metrics import accuracy_score
from peft import LoraConfig, TaskType, get_peft_model, PeftModel, get_peft_model_state_dict
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    return logits.argmax(dim=-1)

def make_deterministic(seed):
    set_seed(seed)
    # torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class SavePeftModelCallback(transformers.TrainerCallback):
    def save_model(self, args, state, kwargs):
        # if state.best_model_checkpoint is not None:
        #     checkpoint_folder = os.path.join(state.best_model_checkpoint, "pt_lora_model")
        # else:
        checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
        
        # if args.enable_peft and args.deepspeed is not None:
        #     logger.info('Peft is enabled, remove deepspeed checkpoint')
        #     shutil.rmtree(os.path.join(checkpoint_folder, f'global_step{state.global_step}'))
        peft_model_path = os.path.join(checkpoint_folder, "pt_lora_model")
        kwargs["model"].save_pretrained(peft_model_path)
        kwargs["tokenizer"].save_pretrained(peft_model_path)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        peft_model_path = os.path.join(args.output_dir, "checkpoint-last")
        peft_model_path = os.path.join(peft_model_path, 'pt_lora_model')
        kwargs["model"].save_pretrained(peft_model_path)
        kwargs["tokenizer"].save_pretrained(peft_model_path)


def accuracy(predictions, references, normalize=True, sample_weight=None):
    return {
        "accuracy": float(
            accuracy_score(references, predictions, normalize=normalize, sample_weight=sample_weight)
        )
    }
        
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # preds have the same shape as the labels, after the argmax(-1) has been calculated
    # by preprocess_logits_for_metrics but we need to shift the labels
    labels = labels[:, 1:].reshape(-1)
    preds = preds[:, :-1].reshape(-1)
    return accuracy(predictions=preds, references=labels)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The tokenizer for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_dir: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    val_data_dir: Optional[str] = field(default=None, metadata={"help": "The name of directory for validation data, if it's not a subset of training data"})
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=True, metadata={"help": "Enable streaming mode"})
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[float] = field(
        default=0.05,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )
    data_cache_dir: Optional[str] = field(default="./", metadata={"help": "The datasets processed stored"})
    supported_languages: Optional[List[str]] = field(default=None, metadata={"help": "The dataset supported language"})
    stage: Optional[List[int]] = field(default_factory=lambda: [-1], metadata={"help": "Training stage defined in description. This controls the data generation"})
    prompter_template: Optional[str] = field(default='alpaca', metadata={"help": "specify prompter template"})
    # sample_num: Optional[int] = field(default=25000, metadata={"help": "Sample ratio from unified dataset"})
    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")


@dataclass
class MyTrainingArguments(TrainingArguments):
    trainable : Optional[str] = field(default="q_proj,v_proj")
    lora_rank : Optional[int] = field(default=8)
    lora_dropout : Optional[float] = field(default=0.1)
    lora_alpha : Optional[float] = field(default=32.)
    modules_to_save : Optional[str] = field(default=None)
    enable_peft : Optional[bool] = field(default=False)
    debug_mode : Optional[bool] = field(default=False)
    peft_path : Optional[str] = field(default=None)
    flash_attn : Optional[bool] = field(default=False)
    need_save: Optional[bool] = field(default=False)
    # val_mode: Optional[str] = field(default='mapping', metadata={"help": "whether use map based validation or "}) 

logger = logging.getLogger(__name__)
if os.environ['LOCAL_RANK'] == 0:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.ERROR)

def main():

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, MyTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if training_args.flash_attn:
        from flash_attn_patch import replace_llama_attn_with_flash_attn
        replace_llama_attn_with_flash_attn()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_clm", model_args, data_args)
    
    # Checking stage
    logger.info("data init stage = {}".format(data_args.stage))
    # Setup logging
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,  # if training_args.local_rank in [-1, 0] else logging.WARN,
        handlers=[logging.StreamHandler(sys.stdout)],)

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    # transformers.tokenization_utils.logging.set_verbosity_warning()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    # set_seed(training_args.seed)
    make_deterministic(training_args.seed)
    logger.warning('deterministic mode is triggered. This may downgrade hardware performance.')

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
        "trust_remote_code": True,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
        "trust_remote_code": True,
        "pad_token": "<unk>"
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.tokenizer_name_or_path:
        tokenizer = LlamaTokenizer.from_pretrained(model_args.tokenizer_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    tokenizer.add_eos_token = False
    
    # Preprocessing the datasets.
    # First we tokenize all the texts.
    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            output = tokenizer(examples["text"])
        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                " before being passed to the model."
            )
        return output
    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                " override this default with `--block_size xxx`."
            )
            block_size = 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    
    with training_args.main_process_first(desc="dataset map tokenization and grouping"):
        lm_datasets = []
        lm_datasets = load_from_disk(data_args.dataset_dir)
        lm_datasets = lm_datasets.filter(lambda entry: entry['mode'] in set(data_args.stage))
        # lm_datasets = lm_datasets.select(np.random.choice(a=len(lm_datasets), size=data_args.sample_num, replace=False)) # allow replacement
        logger.info(f'Unified dataset {data_args.dataset_dir} has been loaded')
        
        if data_args.validation_split_percentage != 0 and data_args.val_data_dir is None and training_args.do_eval:
            logger.warning('Spliting training dataset for validation.')
            # train_ds, val_ds = train_val_dataset(lm_datasets, val_split=data_args.validation_split_percentage)
            # val_single_ds = create_validation_map(val_ds, data_args.supported_languages)
            lm_datasets = lm_datasets.train_test_split(test_size = data_args.validation_split_percentage)
            train_ds, val_single_ds = lm_datasets['train'], lm_datasets['test']
        # elif data_args.validation_split_percentage != 0 and data_args.val_data_dir is not None and training_args.do_eval:
        #     logger.warning('Using {} for evaluation'.format(data_args.val_data_dir))
        #     train_ds = lm_datasets
        #     val_cross_dataset = CrossDataset(data_args.val_data_dir, data_args.supported_languages)
        #     _, val_ds = train_val_dataset(val_cross_dataset, val_split=data_args.validation_split_percentage)
        #     val_single_ds = create_validation_map(val_ds, data_args.supported_languages)
        else:
            logger.warning("No validation is set, adaptive sampling will not be available. Create default spliting")
            train_ds = lm_datasets
          
    # exit()
    # lm_datasets = lm_datasets
    n = len(data_args.supported_languages)
    prompter = Prompter(data_args.prompter_template)
    
    sample_train_collator = MyDataCollator(tokenizer, max_seq_len=block_size, prompter=prompter)
    sample_train_iter = iter(DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=sample_train_collator))
    
    # sample_val_iter = iter(DataLoader(val_ds['en'], batch_size=1, shuffle=False, collate_fn=sample_val_collator))
    if data_args.validation_split_percentage != 0:
        sample_val_collator = MyDataCollator(tokenizer, max_seq_len=block_size, prompter=prompter)
        sample_val_iter = iter(DataLoader(val_single_ds, batch_size=1, shuffle=False, collate_fn=sample_val_collator))
    
    if training_args.do_train:
        train_dataset = train_ds
        # if data_args.max_train_samples is not None:
        #     max_train_samples = min(len(train_dataset), data_args.max_train_samples)
        #     train_dataset = train_dataset.select(range(max_train_samples))
        logger.info(f"Num train_samples  {len(sample_train_iter)}")
        logger.info("Training example:")
        logger.info(tokenizer.decode(next(sample_train_iter)['input_ids'][0]))
    if training_args.do_eval:
        # eval_dataset = val_ds
        eval_dataset = val_single_ds # num_eval_sample * num_lang
        
        # if data_args.max_eval_samples is not None:
        #     max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
        #     eval_dataset = eval_dataset.select(range(max_eval_samples))
        logger.info(f"Num eval_samples  {len(sample_val_iter)}")
        logger.info("Evaluation example:")
        # for sample in sample_val_iter:
        #     logger.info(tokenizer.decode(sample['input_ids'][0]))
        logger.info(tokenizer.decode(next(sample_val_iter)['input_ids'][0]))
    # exit()
    if model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            torch_dtype=torch_dtype,
            trust_remote_code=True
            #low_cpu_mem_usage=True
        )
    else:
        model = AutoModelForCausalLM.from_config(config)
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

    model_vocab_size = model.get_output_embeddings().weight.size(0)
    tokenizer_vocab_size = len(tokenizer)
    logger.info(f"Model vocab size: {model_vocab_size}")
    logger.info(f"Tokenizer vocab size: {tokenizer_vocab_size}")
    if model_vocab_size != tokenizer_vocab_size:
        logger.info(f"Rezize model vocab size to {tokenizer_vocab_size}")
        model.resize_token_embeddings(len(tokenizer))

    if training_args.enable_peft:
        if training_args.peft_path is not None:
            logger.info("Peft from pre-trained model")
            model = PeftModel.from_pretrained(model, training_args.peft_path)
        else:
            logger.info("Init new peft model")
            target_modules = training_args.trainable.split(',')
            modules_to_save = training_args.modules_to_save
            if modules_to_save is not None:
                modules_to_save = modules_to_save.split(',')
            lora_rank = training_args.lora_rank
            lora_dropout = training_args.lora_dropout
            lora_alpha = training_args.lora_alpha
            logger.info(f"target_modules: {target_modules}")
            logger.info(f"lora_rank: {lora_rank}")
            logger.info(f"modues_to_save: {modules_to_save}")
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                target_modules=target_modules,
                inference_mode=False,
                r=lora_rank, lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                modules_to_save=modules_to_save
            )
            model.enable_input_require_grads()
            model = get_peft_model(model, peft_config)
    
    n = len(data_args.supported_languages)
    train_data_collator = MyDataCollator(tokenizer, max_seq_len=block_size, prompter=prompter)
    val_data_collator = MyDataCollator(tokenizer, max_seq_len=block_size, prompter=prompter)
    trainer = CrossLingualTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        train_data_collator=train_data_collator, 
        val_data_collator=val_data_collator if training_args.do_eval else None,
        compute_metrics = compute_metrics if training_args.do_eval else None, #if data_args.sampling_criterion == 'loss' else DecomposeLanguageMetric(supported_langs=data_args.supported_languages, need_save=training_args.need_save), #compute_metrics if training_args.do_eval and not is_torch_tpu_available() else None,
        preprocess_logits_for_metrics = preprocess_logits_for_metrics #if data_args.sampling_criterion == 'loss' else GatherLogits(ignore_index=-100, score_type='close-set')#preprocess_logits_for_metrics
        # if training_args.do_eval and not is_torch_tpu_available()
        # else None,
    )
    
    # trainer.add_callback(BatchSamplingCallback(data_args.sampling_criterion, data_args.supported_languages, train_data_collator, data_args.stage)) # control the stage
    trainer.add_callback(SavePeftModelCallback)
    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

    # # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        
        metrics = trainer.evaluate()

    #     max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
    #     metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
    #     try:
    #         perplexity = math.exp(metrics["eval_loss"])
    #     except OverflowError:
    #         perplexity = float("inf")
    #     metrics["perplexity"] = perplexity

    #     trainer.log_metrics("eval", metrics)
    #     trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
