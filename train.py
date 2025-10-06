import hydra
import torch
import pytorch_lightning as pl
from omegaconf import DictConfig
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
from lightning_module import TranslationGRPOTask
from lightning_data import TranslationDataModule


@hydra.main(version_base=None, config_path="./configs/", config_name="train")
def train(config: DictConfig):
    pl.seed_everything(config.seed, workers=True)

    model, tokenizer = get_model(config)

    illegal_token_mask = None

    data = TranslationDataModule(
        tokenizer=tokenizer,
        illegal_token_mask=illegal_token_mask,
        data_path=config.task.data.path,
        dataset_config_name=config.task.data.dataset_config_name,
        source_lang=config.task.data.source_lang,
        target_lang=config.task.data.target_lang,
        sort_by_length=False
        )
    
    data.setup("fit")

    task = TranslationGRPOTask(
        model=model,
        tokenizer=tokenizer,
        lr=getattr(config.task.training, 'lr', 2e-5),
        num_return_sequences=getattr(config.task.training, 'num_return_sequences', 40),
        max_new_tokens=getattr(config.task.constraints, 'max_sentence_len', 128),
        gen_temperature=getattr(config.task.training, 'gen_temperature', 1.3),
        beta=getattr(config.task.training, 'beta', 0.04),
        clip_param=getattr(config.task.training, 'clip_param', 0.2),
        tgt_lang_id=tokenizer.convert_tokens_to_ids(config.task.data.target_lang),
        reference_model_name=getattr(config.task.model, 'reference_name', None),
    )

    trainer = pl.Trainer(
        accelerator=config.device.accelerator,
        max_epochs=config.task.training.epochs,
        min_epochs=config.task.training.epochs,
        accumulate_grad_batches=config.task.training.accumulate_grad_batches,
        log_every_n_steps=getattr(config.task.training, 'log_every_n_steps', 50),
        check_val_every_n_epoch=10,
        logger=config.logger
        if isinstance(config.logger, bool)
        else hydra.utils.instantiate(config.logger),
        callbacks=[hydra.utils.instantiate(c) for c in config.task.callbacks],
    )

    trainer.fit(model=task, datamodule=data)


def get_model(config: DictConfig):

    # Get the model
    tokenizer = AutoTokenizer.from_pretrained(
        config.task.model.name, 
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        config.task.model.name, attn_implementation="flash_attention_2", dtype=torch.bfloat16
    )
    return model, tokenizer

    


if __name__ == "__main__":
    train()
