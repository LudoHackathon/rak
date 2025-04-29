import hydra
import logging
from omegaconf import DictConfig, OmegaConf

import torch
from torch.utils.data import DataLoader

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import Timer

from base_model import LM
from lit_model import LitLM

from data import DummyDataset


log = logging.getLogger(__name__)

@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    """
    Evaluate train and inference time
    """
    seed_everything(42, workers=True)

    ########
    # Data #
    ########

    dataset = hydra.utils.instantiate(cfg.data)  # we use the same dummy dataset for train and val
    if cfg.get("train"):
        train_dataloader = DataLoader(dataset, batch_size=cfg.train_bs, num_workers=14)  # no need shuffling with our dummydataset
    val_dataloader = DataLoader(dataset, batch_size=cfg.val_bs, num_workers=14)

    #########
    # Model #
    #########

    model = hydra.utils.instantiate(cfg.model)

    if cfg.compile:
        model = torch.compile(model)


    ###########
    # Trainer #
    ###########

    timer = Timer()
    trainer = Trainer(
        deterministic=True,
        callbacks=timer,
        max_epochs=3,
    )

    ###########################
    # Mode: Train/Val or Test #
    ###########################

    if cfg.get("train"):
        trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
        log.info(f"{cfg.task_name} train time: {timer.time_elapsed('train')}")
        log.info(f"{cfg.task_name} val time: {timer.time_elapsed('validate')}")
    else:
        trainer.test(model, dataloaders=val_dataloader)
        log.info(f"{cfg.task_name} test time: {timer.time_elapsed('test')}")


if __name__ == "__main__":
    main()
