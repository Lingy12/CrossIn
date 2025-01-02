from typing import List, Sequence, Dict
from datasets import Dataset as HFDataset
from torch.utils.data import Dataset, Subset, DataLoader
import os
import json
import logging
import sys
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from transformers import PreTrainedTokenizer
import torch
import random
from scipy.stats import norm
from cross_lingual_utils.prompter import Prompter
import numpy as np
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,  # if training_args.local_rank in [-1, 0] else logging.WARN,
        handlers=[logging.StreamHandler(sys.stdout)],)

logger = logging.getLogger(__name__)
# if os.environ['LOCAL_RANK'] == 0:
#     logger.setLevel(logging.DEBUG)
# else:
#     logger.setLevel(logging.ERROR)


# B_INST, E_INST = "[INST]", "[/INST]"
# B_SYS, E_SYS = "<<SYS>>", "<</SYS>>\n\n"
# SYS_PROMPT = 'You are a helpful assistant.'
IGNORE_INDEX=-100

def train_val_dataset(dataset, val_split=0.25):
    if val_split == 1:
        return dataset, dataset
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    return Subset(dataset, train_idx), Subset(dataset, val_idx)

def create_validation_map(val_datasets, supported_language):

    data_loader = DataLoader(val_datasets, batch_size=None, batch_sampler=None)
    # import pdb; pdb.set_trace() 
    dataset_dict = {lang: {"instruction": [], "input": [], "output": []} for lang in supported_language}

    for entry in iter(data_loader):
        for lang in entry.keys():
            for k in dataset_dict[lang].keys():
                dataset_dict[lang][k].append(entry[lang][k])
       
    single_dataset = {"instruction": [], "input": [], "output": []}
    for lang in dataset_dict:
        dataset_dict[lang] = HFDataset.from_dict(dataset_dict[lang]) # mapping version
        for k in single_dataset:
            single_dataset[k].extend(dataset_dict[lang][k]) # need to ensure the order of the dataset
    
    return HFDataset.from_dict(single_dataset)


class CrossDataset(Dataset):
    '''
    This class is used to create a cross lingual dataset. 

    All files under data_folder must have json format, with same amount of data.
    '''
    def __init__(self, data_folder: str, supported_lang_suffixes: List[str]):
        self.data_folder = os.path.normpath(data_folder)
        self.supported_lang_suffixes = supported_lang_suffixes

        data_json_map = {}
        for lang in self.supported_lang_suffixes:
            with open(os.path.join(data_folder, os.path.basename(self.data_folder) + '_' + lang + '.json'), 'r') as f:
                data_json_map[lang] = json.load(f)
       
        self.data = []
        
        for i in range(len(data_json_map[self.supported_lang_suffixes[0]])):
            entry = {}
            for lang in self.supported_lang_suffixes:
               entry[lang] = data_json_map[lang][i]
            self.data.append(entry)

        logger.info('Cross lingual dataset from {} is initialized.'.format(data_folder))
        logger.info('Sample data: ')
        logger.info(self.data[0])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

# def create_probability_vector(supported_lang, criterion_name, metrics):
    
    
# This collator is used for training (archived)
@dataclass
class AdaptiveDataCollator(object):
     
    tokenizer: PreTrainedTokenizer
    probability: List[float]
    supported_lang: List[str]
    max_seq_len: int
    stage: int # 0, 1, 2, 3
    prompter: Prompter
    _criterion: List[Dict[str, float]] = None # list of criterion

    '''
    -1: English only
    0: No updating probability (matching input output)
    1: Probability updating (matching input output, confidence score) naive choose
    2: Random left right (different input output)
    3: 1 combine with 2. Use probability select language to align with English in input by using translation pair, random select right side
    '''

    def update_criterion(self, criterion):
        if not self._criterion:
            self._criterion = []
        self._criterion.append(criterion)
        logger.info("Data collator receive new criterion = {}".format(criterion))
        self._update_prob_vec()
    
    # The probability is proportional to the criterion
    def _criterion_to_prob(self, criterion_map, scale=1):
        raw_criterion_score = np.array([criterion_map[lang] * scale for lang in self.supported_lang])
        # exp_score = np.exp(raw_criterion_score)
        z_score = (raw_criterion_score - raw_criterion_score.mean()) / raw_criterion_score.std()
        scaled_score = norm.cdf(z_score)
        logger.info("scaled score = {}".format(scaled_score))
        return scaled_score / np.sum(scaled_score)

    def _update_prob_vec(self):
        if len(self._criterion) == 0:
            raise Exception("You shouldn't update probability when no criterion appended.")
        if self.stage == 1 or self.stage == 3:
            logger.info("Collator creating new probability with latest confidence score")
            new_prob = self._criterion_to_prob(self._criterion[-1])
            logger.info(new_prob)
            self.probability = new_prob
        else:
            logger.info(f"Not updating because stage is {self.stage}")
            return
        logger.info("New probability assigned")


    def update_stage(self, stage):
        self.probability = [1/len(self.supported_lang) for _ in range(len(self.supported_lang))]
        logger.info("Probability reset when updating stage")
        self.stage = stage

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        logger.info("Collating with probability: {}".format(self.probability))
        logger.info("Collator stage: {}".format(self.stage))
        assert len(self.probability) == len(self.supported_lang)
        batch_size = len(instances)
        
        # decide input output language
        if self.stage == -1:
            input_langs = ["en"] * batch_size
        else:
            input_langs = np.random.choice(self.supported_lang, size=batch_size, replace=True, p=self.probability)
        
        if self.stage == 2:
            output_langs = np.random.choice(self.supported_lang, size=batch_size, replace=True, p=self.probability)
        elif self.stage == 3:
            output_langs = np.random.choice(self.supported_lang, size=batch_size, replace=True, p=[1/len(self.supported_lang) for _ in range(len(self.supported_lang))]) # random select output pair
        else:
            output_langs = input_langs
  

        instructions, inputs = tuple([instances[i][input_langs[i]][key] for i in range(len(input_langs))] 
                                              for key in ('instruction', 'input'))
        
        # collate english data
        if self.stage == 3:
            en_instructions, en_inputs = tuple([instance['en'][key] for instance in instances] for key in ('instruction', 'input'))
            
        outputs = [instances[i][output_langs[i]]['output'] for i in range(len(output_langs))] 

        sources, targets = [], []

        # need translation pair
        if self.stage == 3:
            logger.info("Adding translation pair")
            for instruction, input, output,en_instruction, en_input, input_lang, output_lang in zip(instructions, inputs, outputs, en_instructions, en_inputs, input_langs, output_langs):
                # if input is not None and input != '':
                #     instruction = instruction + '\n' + input
                    # en_instruction = en_instruction + '\n' + en_input
                # logger.info(instruction)
                # else:
                # if input is None or input != 
                if input_lang != 'en':
                    instruction = f"[{input_lang}] {instruction} \n [en] {en_instruction}"
                    input = f"[{input_lang}] {input} \n [en] {en_input}" if input != '' and input is not None else None
                    # sources.append(f'{B_INST} {B_SYS}\n {SYS_PROMPT} {E_SYS} [{input_lang}] {instruction} \n [en] {en_instruction} {E_INST}')
                else:
                    instruction = f"{instruction}"
                    input = f"{input} \n" if input != '' and input is not None else None
                    output_lang = ''
                    # sources.append(f'{B_INST} {B_SYS}\n {SYS_PROMPT} {E_SYS} [{input_lang}] {instruction} {E_INST}') # don't add redaundant tokens for english
                sources.append(self.prompter.generate_prompt(instruction=instruction, input=input, output_lang=f'[{output_lang}]' if len(output_lang) > 0 else ''))
                targets.append(f"{output}{self.tokenizer.eos_token}")
        else:
            # collater without translation pair
            for instruction, input, output, input_lang, output_lang in zip(instructions, inputs, outputs, input_langs, output_langs):
                # if input is not None and input != '':
                #     instruction = instruction + '\n' + input
                if self.stage == -1:
                    instruction = f"{instruction}"
                    input = f"{input}" if input != '' and input is not None else None
                    output_lang = ''
                else:
                    instruction = f"[{input_lang}] {instruction}"
                    input = f"[{input_lang}] {input}" if input != '' and input is not None else None
                sources.append(self.prompter.generate_prompt(instruction=instruction, input=input, output_lang=f'[{output_lang}]' if len(output_lang) > 0 else '')) # make the tag here
                targets.append(f"{output}{self.tokenizer.eos_token}")
                
        tokenized_sources = self.tokenizer(sources, return_attention_mask=False)
        tokenized_target = self.tokenizer(targets, return_attention_mask=False, add_special_tokens=False)

        all_input_ids = []
        all_labels = []

        for s,t in zip(tokenized_sources['input_ids'],tokenized_target['input_ids']):
            input_ids = torch.LongTensor(s + t)[:self.max_seq_len]
            labels = torch.LongTensor([IGNORE_INDEX] * len(s) + t)[:self.max_seq_len]
            assert len(input_ids) == len(labels)
            all_input_ids.append(input_ids)
            all_labels.append(labels)
        
        input_ids = torch.nn.utils.rnn.pad_sequence(
                all_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
                )
        targets = torch.nn.utils.rnn.pad_sequence(all_labels, batch_first=True, padding_value=IGNORE_INDEX)

        return dict(
                input_ids = input_ids,
                labels = targets
                )

@dataclass
class MyDataCollator(object):
     
    tokenizer: PreTrainedTokenizer
    max_seq_len: int    
    prompter: Prompter

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        instructions, inputs, outputs = tuple([instance[key] for instance in instances] for key in ('instruction', 'input', 'output'))
        # print(instructions[0])
        # print(inputs[0])
        # print(targets[0])
        sources, targets = [], []
        # print(all_data[0])
        # import pdb; pdb.set_trace()
        for instruction, input, output in zip(instructions, inputs, outputs):
            # print(instruction)
            # if input is not None and input != '':
            #     instruction = instruction + '\n' + input
            # sources.append(f'{B_INST} {B_SYS}\n {SYS_PROMPT} {E_SYS} {instruction} {E_INST}')
            input = f"{input}" if input != '' and input is not None else None
            sources.append(self.prompter.generate_prompt(instruction=instruction, input=input, output_lang='')) # make the tag here
            targets.append(f"{output}{self.tokenizer.eos_token}")
        # print(sources[0]) 
        tokenized_sources = self.tokenizer(sources, return_attention_mask=False)
        tokenized_target = self.tokenizer(targets, return_attention_mask=False, add_special_tokens=False)

        all_input_ids = []
        all_labels = []

        for s,t in zip(tokenized_sources['input_ids'],tokenized_target['input_ids']):
            input_ids = torch.LongTensor(s + t)[:self.max_seq_len]
            labels = torch.LongTensor([IGNORE_INDEX] * len(s) + t)[:self.max_seq_len]
            assert len(input_ids) == len(labels)
            all_input_ids.append(input_ids)
            all_labels.append(labels)
        
        input_ids = torch.nn.utils.rnn.pad_sequence(
                all_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
                )
        targets = torch.nn.utils.rnn.pad_sequence(all_labels, batch_first=True, padding_value=IGNORE_INDEX)

        return dict(
                input_ids = input_ids,
                labels = targets
                )


