from types import MethodType
import hydra
import torch
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
from utils import (
    FrozenModelSentenceGivenPrompt,
    ReplayBuffer,
)
from lightning_module import TranslationGFNTask
from lightning_data import TranslationDataModule


@hydra.main(version_base=None, config_path="./configs/", config_name="train")
def train(config: DictConfig):
    pl.seed_everything(config.seed, workers=True)

    model, tokenizer = get_model(config)

    end_of_sentence_token_id = tokenizer.eos_token_id
        
    illegal_token_mask = None
    reward = get_reward(config, end_of_sentence_token_id, illegal_token_mask)
    reward_buffer = ReplayBuffer(
        buffer_size=config.task.reward.buffer_size,
        termination_token_id=end_of_sentence_token_id,
    )

    data = TranslationDataModule(
        tokenizer=tokenizer,
        illegal_token_mask=illegal_token_mask,
        data_path=config.task.data.path,
        dataset_config_name=config.task.data.dataset_config_name,
        source_lang=config.task.data.source_lang,
        target_lang=config.task.data.target_lang,
        )
    
    data.setup("fit")
    train_probes = [(data.train_data[i][0][0], data.train_data[i][1]) for i in range(config.task.eval.n_probes)]
    val_probes = [(data.val_data[i][0][0], data.val_data[i][1]) for i in range(config.task.eval.n_probes)]

    task = TranslationGFNTask(
        model=model,
        tokenizer=tokenizer,
        reward=reward,
        reward_buffer=reward_buffer,
        n_samples=config.task.training.n_samples,
        lr=config.task.training.lr,
        subtb_lambda=config.task.training.subtb_lambda,
        pf_temp_high=config.task.training.pf_temp_high,
        pf_temp_low=config.task.training.pf_temp_low,
        pf_temp_prob=config.task.training.pf_temp_prob,
        use_buffer_prob=config.task.training.use_buffer_prob,
        min_sentence_len=config.task.constraints.min_sentence_len,
        max_sentence_len=config.task.constraints.max_sentence_len,
        reward_temp_start=config.task.reward.temp_start,
        reward_temp_end=config.task.reward.temp_end,
        reward_temp_horizon=config.task.reward.temp_horizon,
        illegal_token_mask=illegal_token_mask,
        train_probes=train_probes,
        val_probes=val_probes,
        tgt_lang_id=tokenizer.convert_tokens_to_ids(config.task.data.target_lang),
    )

    trainer = pl.Trainer(
        accelerator=config.device.accelerator,
        max_epochs=config.task.training.epochs,
        accumulate_grad_batches=config.task.training.accumulate_grad_batches,
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
        config.task.model.name, attn_implementation="flash_attention_2", dtype=torch.float16
    )
    return model, tokenizer


def get_reward(config: DictConfig, sentence_token_id, illegal_token_mask):
    if config.task.reward.sentence_validator is None:
        sentence_validator, valid_sentence_alpha = None, None
    elif config.task.reward.sentence_validator == "rule":
        sentence_validator, valid_sentence_alpha = (
            # RuleSentenceValidator(sentence_token_id=sentence_token_id),
            config.task.reward.valid_sentence_alpha,
        )
    elif config.task.reward.sentence_validator == "model":
        # sentence_validator, valid_sentence_alpha = (
        #     ModelSentenceValidator(sentence_token_id=sentence_token_id),
        #     config.task.reward.valid_sentence_alpha,
        # )
        pass
    else:
        raise ValueError(
            f"Invalid sentence validator: {config.task.reward.sentence_validator}"
        )

    reward = FrozenModelSentenceGivenPrompt(
        sentence_token_id=sentence_token_id,
        min_len=config.task.constraints.min_sentence_len,
        vocab_alpha=config.task.reward.vocab_alpha,
        vocab_naughty_mask=illegal_token_mask,
        sentence_validator=sentence_validator,
        valid_sentence_alpha=valid_sentence_alpha,
    )

    return reward


if __name__ == "__main__":
    train()
