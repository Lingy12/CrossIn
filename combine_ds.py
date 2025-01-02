import fire
from datasets import load_from_disk
from datasets import concatenate_datasets
from typing import List

def combine_ds(ds_lst: List[str], output_name):
    dss = []

    for ds in ds_lst:
        print(ds)
        dss.append(load_from_disk(ds))
        print(dss[-1])

    output_ds = concatenate_datasets(dss)

    output_ds.save_to_disk(output_name)

if __name__ == "__main__":
    fire.Fire(combine_ds)
