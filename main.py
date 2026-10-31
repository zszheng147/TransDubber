import gc
import os

import hydra
from omegaconf import DictConfig, OmegaConf

import torch
import lightning as L
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from data import LitDataModule
from model import LitModelModule


@hydra.main(config_path="config", config_name="s2s-stage1-asr-ast", version_base=None)
def main(config: DictConfig):
    L.seed_everything(config.common.seed)

    exp_name = f"{config.common._name}"
    # Use a stable ID to resume logging with the same training run.
    wandb_logger = None if exp_name == "debug" else \
        L.pytorch.loggers.WandbLogger(
            project="TransDubber", name=exp_name,
            id=exp_name, resume="allow",
        )

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{config.common.output_dir}/checkpoints", 
        save_top_k=10, monitor="val_loss", save_last=True
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_monitor]

    model = LitModelModule(config)
    datamodule = LitDataModule(config=config.dataset, processor=model.processor)

    if config.checkpoint.resume_ckpt_path:
        checkpoint = torch.load(
            config.checkpoint.resume_ckpt_path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        msg = model.load_state_dict(checkpoint["state_dict"], strict=False)
        print("Loaded checkpoint from ", config.checkpoint.resume_ckpt_path, "with msg: ", msg)
        del checkpoint
        gc.collect()

    precision = "bf16-mixed" if model.model_dtype == torch.bfloat16 else "16-mixed"
    print(
        "Training precision: ", precision,
        "; parameter dtype: ", model.train_param_dtype,
        flush=True,
    )

    trainer = L.Trainer(
        num_nodes=config.common.num_nodes,
        devices="auto",
        accelerator="gpu",
        precision=precision,
        strategy=DDPStrategy(find_unused_parameters=False),
        accumulate_grad_batches=config.optimization.accumulate_grad_batches,
        use_distributed_sampler=False, # set to False if you use specific sampler
        num_sanity_val_steps=0,
        fast_dev_run=False,
        val_check_interval=config.checkpoint.val_check_interval,
        check_val_every_n_epoch=config.checkpoint.check_val_every_n_epoch,
        max_epochs=config.optimization.max_epoch,
        max_steps=config.optimization.max_steps,
        gradient_clip_val=1.0,
        logger=wandb_logger,
        log_every_n_steps=config.optimization.log_steps,
        enable_checkpointing=True,
        callbacks=callbacks,
    )

    trainer.fit(model, datamodule=datamodule, ckpt_path="last")


if __name__ == "__main__":
    main()
