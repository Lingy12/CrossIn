import torch
import torch.nn as nn
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
import logging
from cross_lingual_utils.cross_datasets import AdaptiveDataCollator
from typing import Dict, List
import sys
import numpy as np
import os

from dataclasses import dataclass

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,  # if training_args.local_rank in [-1, 0] else logging.WARN,
        handlers=[logging.StreamHandler(sys.stdout)],)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# if os.environ['LOCAL_RANK'] == 0:
#     logger.setLevel(logging.DEBUG)
# else:
#     logger.setLevel(logging.ERROR)
@dataclass
class GatherLogits:
    '''
    This function gathers the logits from raw prediction to token id. (B, S, V) -> (B, S)
    '''
    ignore_index: int = -100
    score_type: str = 'close-set'

    def __call__(self, logits, labels):
        logits = logits[..., :-1, :].contiguous()
        labels = labels[..., 1:].contiguous()

        log_probs = -nn.functional.log_softmax(logits, dim=-1)
        if labels.dim() == log_probs.dim() - 1:
            labels = labels.unsqueeze(-1)
        # print(log_probs.shape, labels.shape)
        # In case the ignore_index is -100, the gather will fail, so we replace labels by 0. The padding_mask
        # will ignore them in any case.
        
        labels = torch.clamp(labels, min=0)
        
        if self.score_type == 'close-set':
            token_prob = log_probs.gather(dim=-1, index=labels)
        elif self.score_type == 'open-generation':
            # logger.info(log_probs.shape, torch.argmax(log_probs, dim=-1).shape)
            # print(labels.shape)
            # print(log_probs.shape, torch.argmax(log_probs, dim=-1, keepdim=True).shape)
            token_prob = log_probs.gather(dim=-1, index=torch.argmax(log_probs, dim=-1, keepdim=True))
        else:
            raise Exception("confidence score type is not implemented. Please use close-set or open-generation")
        # print('shape in gather:')
        # print(gathered_logits.shape, labels.shape)
        return token_prob, labels
    
@dataclass
class ConfidenceScore :
    """
    Adapted from.
    https://github.com/huggingface/transformers/blob/v4.36.1/src/transformers/trainer_pt_utils.py#L480
    Adds label-smoothing on a pre-computed output from a Transformers model.

    Open generation score only cares about generated logits, but we need to label to 
    calculate the mask
    """
    
    ignore_index: int = -100
    # score_type: str = "close-set" # close-set or open-generation https://arxiv.org/pdf/2305.14802.pdf
    
    def __call__(self, token_prob, labels):
        # logits = model_output["logits"] if isinstance(model_output, dict) else model_output[0]
        # trick to force cpu eval to prevent OOM during evaluation. logits and labels are np array when using eval_accumulation_steps
        token_prob, labels = torch.tensor(token_prob), torch.tensor(labels)
        # print(token_prob.shape, labels.shape)
        padding_mask = labels.eq(self.ignore_index)

        # works for fp16 input tensor too, by internally upcasting it to fp32
        # smoothed_loss = log_probs.sum(dim=-1, keepdim=True, dtype=torch.float32)

        token_prob.masked_fill_(padding_mask, 0.0)
        # smoothed_loss.masked_fill_(padding_mask, 0.0)

        # Take the mean over the label dimensions, then divide by the number of active elements (i.e. not-padded):
        num_active_elements = padding_mask.numel() - padding_mask.long().sum()
        confidence_score = token_prob.sum() / num_active_elements
        # smoothed_loss = smoothed_loss.sum() / (num_active_elements * log_probs.shape[-1])
        # print(token_prob.sum(dim=1).shape, padding_mask.shape)
        return {"overall_confidence": confidence_score, "sample_confidence": token_prob.sum(dim=1) / (padding_mask.shape[1] - padding_mask.long().sum(dim=1))}

@dataclass
class DecomposeLanguageMetric(object):
    supported_langs: List[str]
    need_save: bool = False
    def __call__(self, eval_preds):
        logger.info("Decompose language metric")
        logger.info('Compute metric ')
        gathered_logits, _ = eval_preds
        shifted_token_prob, shifted_label = gathered_logits
        # print(shifted_token_prob.shape)
        assert len(shifted_token_prob) % len(self.supported_langs) == 0 and len(shifted_label) % len(self.supported_langs) == 0
        confidence_score = ConfidenceScore(ignore_index=-100)
        splitted_prob, splitted_label = np.array_split(shifted_token_prob, len(self.supported_langs)), np.array_split(shifted_label, len(self.supported_langs))
        result_map = {self.supported_langs[i]: (splitted_prob[i], splitted_label[i]) for i in range(len(self.supported_langs))}
        
        overall_confidence = {}
        sample_wise_confidence = []
        
        for lang in self.supported_langs:
            confidence_res = confidence_score(result_map[lang][0], result_map[lang][1])
            overall_confidence[lang] = confidence_res['overall_confidence']
            sample_wise_confidence.append(confidence_res['sample_confidence'])
        
        raw_sample_score = torch.cat(tuple(sample_wise_confidence), dim=1) # assert the first is reference language, i.e. english, we compare all with first raw
        z_sample = raw_sample_score - torch.mean(raw_sample_score, dim=0) / torch.std(raw_sample_score, dim=0)
        # print(raw_sample_score)
        # print(z_sample)
        # print(raw_sample_score.shape)
        corr_matrix = torch.corrcoef(z_sample.T)

        if self.need_save:
            os.makedirs('./manifest', exist_ok=True)
            torch.save(z_sample.T, './manifest/confidence_z_series.pt')
            logger.info('raw series saved.')
        # print(torch.corrcoef(z_sample.T)) # z_sample shape = (sample_num, num_lang) so we transpose
        language_confidence =  {
            f"{lang}_confidence": overall_confidence[lang] for lang in self.supported_langs
        }
        
        language_corr = {
            f"{self.supported_langs[i]}_corr": corr_matrix[0][i] for i in range(len(self.supported_langs)) # force proportional, smaller -> smaller
        }
        language_confidence.update(language_corr)
        
        return language_confidence
@dataclass
class BatchSamplingCallback(TrainerCallback):
    
    criterion_name: str
    supported_languages: List[str]
    adaptive_collator: AdaptiveDataCollator
    stages: List[int]
    
    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
       
        # if state.is_local_process_zero:
        if "eval_criterion_dict" in dir(state) and len(state.eval_criterion_dict) > 0:
            logger.info("Eval criterion dict = {}".format(state.eval_criterion_dict))
            # new_prob = criterion_to_prob(state.eval_criterion_dict, self.supported_languages)
            # logger.info("New probability = {}".format(new_prob))
            self.adaptive_collator.update_criterion(state.eval_criterion_dict)
            logger.info("Eval criterion reset.")
        else:
            logger.info("Not updating sampling this step.")
        
        state.eval_criterion_dict = {}
        return super().on_step_begin(args, state, control, **kwargs)
    
    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if len(self.stages) > 1:
            logger.info(self.stages)
            logger.info("Update collator stage at epoch = {} with stage = {}".format(state.epoch, self.stages[round(state.epoch)]))
            self.adaptive_collator.update_stage(self.stages[round(state.epoch)]) # rounding down float
        else:
            logger.info("Not changing stage")
            
        return super().on_epoch_begin(args, state, control, **kwargs)
    
    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics: Dict[str, float], **kwargs):
        # if state.is_local_process_zero:
        logger.info("Triggered batch sampling callback")
        logger.info("metric = {}".format(metrics))
        # logger.info("target criterion = {}",format(self.criterion_name[]))
        # logger.info(self.criterion_name)
        for k in metrics.keys():
            # logger.info(k.split('_'))
            if len(k.split('_')) == 3:
                _, lang, criterion = k.split('_')
                logger.info(criterion)
                if criterion == self.criterion_name:
                    state.eval_criterion_dict[lang] = metrics[k]
        if len(state.eval_criterion_dict) == 0:
            raise Exception("Cannot find matching criterion")
            
        
