from typing import Optional
from torch.utils.data import DataLoader, Dataset
from transformers import Trainer, DataCollator
import torch

class CrossLingualTrainer(Trainer):
    
    def __init__(self, train_data_collator: DataCollator, val_data_collator: DataCollator, **kwargs):
        super().__init__(**kwargs)
        self.train_data_collator = train_data_collator
        self.val_data_collator = val_data_collator
    
    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        # data_collator = self.data_collator # Do not use default collator
        # if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
        #     train_dataset = self._remove_unused_columns(train_dataset, description="training")
        # else:
        #     data_collator = self._get_collator_with_removed_columns(data_collator, description="training")
        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": self.train_data_collator, # use train data collator
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "shuffle": True,
            "persistent_workers": True if self.args.dataloader_num_workers > 1 else False,
            "drop_last": True
        }

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))
    
    def get_eval_dataloader(self, eval_dataset: Dataset | None = None) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (`torch.utils.data.Dataset`, *optional*):
                If provided, will override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns not accepted
                by the `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": self.val_data_collator, # use val data collator
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory
        }

        # if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
        #     dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)
        #     dataloader_params["drop_last"] = self.args.dataloader_drop_last

        return self.accelerator.prepare(DataLoader(eval_dataset, **dataloader_params))