import torch
from torch.utils.data import DataLoader, Sampler
from torchdata.datapipes.map import MapDataPipe
from pytorch_lightning import LightningDataModule
import warnings
from datasets import load_dataset

warnings.filterwarnings("ignore", ".*does not have many workers.*")


class LengthSortedSampler(Sampler[int]):
    def __init__(self, lengths, ascending: bool = True):
        self.lengths = lengths
        self.indices = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=not ascending)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class TranslationDataModule(LightningDataModule):
    def __init__(
        self,
        tokenizer,
        illegal_token_mask,
        data_path,
        dataset_config_name,
        source_lang,
        target_lang,
        sort_by_length: bool = True,
        sort_direction: str = "asc",
        train_batch_size: int = 1,
    ):
        super().__init__()
        self.save_hyperparameters(ignore="tokenizer")
        self.tokenizer = tokenizer
        self.train_data = None
        self.val_data = None
        self._train_sampler = None
        self._train_batch_size = max(int(train_batch_size), 1)

    def setup(self, stage):
        prompts = load_dataset(
            self.hparams.data_path, self.hparams.dataset_config_name, trust_remote_code=True
        )
        
        self.train_data = TranslationDataPipe(prompts["train"], self.tokenizer, src_col="sentence_" + self.hparams.source_lang, tgt_col="sentence_" + self.hparams.target_lang)
        self.val_data = TranslationDataPipe(prompts["valid"], self.tokenizer, src_col="sentence_" + self.hparams.source_lang, tgt_col="sentence_" + self.hparams.target_lang)

        # Optional: build a sampler to iterate by increasing source length
        if self.hparams.sort_by_length:
            src_col = "sentence_" + self.hparams.source_lang
            # Compute source lengths once
            lengths = [
                len(self.tokenizer(self.train_data.prompts[i][src_col]).input_ids)  # type: ignore[attr-defined]
                for i in range(len(self.train_data))
            ]
            ascending = (self.hparams.sort_direction.lower() != "desc")
            self._train_sampler = LengthSortedSampler(lengths, ascending=ascending)

    def train_dataloader(self):
        if self._train_sampler is not None:
            return DataLoader(
                self.train_data,
                sampler=self._train_sampler,
                batch_size=self._train_batch_size,
                num_workers=0,
                collate_fn=self._collate_batch,
            )
        return DataLoader(
            self.train_data,
            shuffle=True,
            batch_size=self._train_batch_size,
            num_workers=0,
            collate_fn=self._collate_batch,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=48,
            num_workers=0,
            collate_fn=self._collate_batch,
        )

    def _collate_batch(self, batch):
        encoder_inputs_list, targets, sample_ids = zip(*batch)
        prepared_inputs = []
        for inputs in encoder_inputs_list:
            item = {}
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    squeezed = value.squeeze(0)
                    item[key] = squeezed.tolist() if squeezed.ndim == 1 else squeezed
                else:
                    item[key] = value
            prepared_inputs.append(item)
        batch_encoding = self.tokenizer.pad(
            prepared_inputs,
            padding=True,
            return_tensors="pt",
        )
        return batch_encoding, list(targets), list(sample_ids)


class TranslationDataPipe(MapDataPipe):
    def __init__(self, prompts, tokenizer, src_col, tgt_col) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.prompts = prompts
        self.src_col = src_col
        self.tgt_col = tgt_col

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, index):
        src_prompt = self.tokenizer(
            self.prompts[index][self.src_col],
            return_tensors="pt",
        )
        str_tgt_prompt = self.prompts[index][self.tgt_col]
        return src_prompt, str_tgt_prompt, index
