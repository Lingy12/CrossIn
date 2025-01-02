import fire
import os
import numpy as np
from datasets import load_dataset, concatenate_datasets, Dataset
import pandas as pd
import statsmodels.api as sm
import torch
from typing import List, Dict
from scipy.stats import norm
import shutil

np.random.seed(0)
    
def build_complete_dataset(ori_folder:str, output_path:str, supported_languages: List = ['en', 'zh', 'vi', 'es'], add_sample: int = 1500):
    # return
    # if not os.path.exists(os.path.basename(output_path)):
    #     os.makedirs(os.path.basename(output_path))
    
    # calculate correlation for each language and sample weight
    # data = torch.load(z_score_tensor_path)
    # print(data.shape)
    # first row is english
    if os.path.exists(output_path):
        # print('Your intended path already exists, do you want to remove?')
        flag = input('Your intended path already exists, do you want to remove?')
        if flag == 'y':
            shutil.rmtree(output_path)
        else:
            return ;

    # data = pd.DataFrame({supported_languages[idx]: data[idx] for idx in range(len(supported_languages))})
    
    # build mode = 0: concat all
    print('mode = 0')
    ds_map = {}
    ds_map_complete = {}
    for lang in supported_languages:
        ds = load_dataset(ori_folder, config=lang, split='train')
        ds = ds.add_column('language', [lang] * len(ds))
        ds_map_complete[lang] = ds
        target_length = len(ds)
        if lang != 'en' and add_sample != -1:
            target_length = add_sample
        
        ds = ds.select(np.random.choice(a=len(ds), size=target_length, replace=False)) # select required sample
        # ds = ds.add_column('mode', ['0'] * len(ds))
        ds_map[lang] = ds
        # print(ds)
    mode_0_ds = concatenate_datasets(ds_map.values())
    mode_0_ds = concatenate_datasets([mode_0_ds, Dataset.from_dict({"mode": [0] * len(mode_0_ds)})], axis=1)
    
    p = np.array([0] + [1/(len(supported_languages) - 1)] * (len(supported_languages) - 1))
    language_selection = np.random.choice(supported_languages, 
                                          size=add_sample if add_sample > 0 else len(ds_map['en']), replace=True, p=p) # add probability vector
    assert all(language_selection != 'en')
    # sample_selection = np.random.choice([], size=num_samples, replace=True) # select top k 

    num_sample = add_sample if add_sample > 0 else len(ds_map['en'])
    mode_1_dict = {"input": [], "output": [], "instruction": [], "mode": [1] * num_sample, "language": ["cross"] * num_sample}
    mode_2_dict = {"input": [], "output": [], "instruction": [], "mode": [2] * num_sample, "language": ["cross"] * num_sample} 
    mode_3_dict = {"input": [], "output": [], "instruction": [], "mode": [3] * num_sample, "language": ["cross"] * num_sample} 
    # build input output pair data

    for i in range(num_sample):
        sample_idx = int(np.random.choice(len(ds_map_complete['en'])))
        # print(sample_idx)
        en_entry = ds_map_complete['en'][sample_idx]
        target_lang_entry = ds_map_complete[language_selection[i]][sample_idx]
        # data_source, id = en_entry['data_source'], en_entry['id']

        mode_1_dict['input'].append(en_entry['input'] + '\n' + f'[{language_selection[i]}]')
        mode_1_dict['instruction'].append(en_entry['instruction'])
        mode_1_dict['output'].append(target_lang_entry['output'])
        # mode_1_dict['data_source'].append(data_source)
        # mode_1_dict['id'].append(id)
        
        mode_2_dict['input'].append(target_lang_entry['input'] + '\n' + '[en]')
        mode_2_dict['instruction'].append(target_lang_entry['instruction'])
        mode_2_dict['output'].append(en_entry['output'])
        # mode_2_dict['data_source'].append(data_source)
        # mode_2_dict['id'].append(id)
        
        mode_3_dict['input'].append(target_lang_entry['instruction'])
        mode_3_dict['instruction'].append('Translate the following sentence into English.')
        mode_3_dict['output'].append(en_entry['instruction'])
        # mode_3_dict['data_source'].append(data_source)
        # mode_3_dict['id'].append(id)
        
    mode_1_ds = Dataset.from_dict(mode_1_dict)
    mode_2_ds = Dataset.from_dict(mode_2_dict)
    mode_3_ds = Dataset.from_dict(mode_3_dict)
    
    print(mode_1_ds[-1])
    print(mode_2_ds[-1])
    print(mode_3_ds[-1])
    
    overall_ds = concatenate_datasets([mode_1_ds, mode_2_ds, mode_3_ds])
    # overall_ds = mode_0_ds
    
    
    print(len(overall_ds))
    overall_ds.save_to_disk(output_path)
    
    
    

if __name__ == '__main__':
    fire.Fire(build_complete_dataset)
