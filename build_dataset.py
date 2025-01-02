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
    for lang in supported_languages:
        ds = load_dataset(ori_folder, config=lang, split='train')
        target_length = len(ds)
        if lang != 'en' and add_sample != -1:
            target_length = add_sample
        
        ds = ds.select(np.random.choice(a=len(ds), size=target_length, replace=False)) # select required sample
        # ds = ds.add_column('mode', '0')
        ds = ds.add_column('language', [lang] * len(ds))
        ds_map[lang] = ds
        # print(ds)
    mode_0_ds = concatenate_datasets(ds_map.values())
    mode_0_ds = concatenate_datasets([mode_0_ds, Dataset.from_dict({"mode": [0] * len(mode_0_ds)})], axis=1)
    
    overall_ds = mode_0_ds
    print(len(overall_ds))
    overall_ds.save_to_disk(output_path)

if __name__ == '__main__':
    fire.Fire(build_complete_dataset)
