from pytorch_lightning import Trainer, profiler, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping
import torch, os
from torch.utils.data import DataLoader
from torchvision import models
from absl import app, flags
from ml_collections.config_flags import config_flags
from ml_collections import ConfigDict
from dataset import CovidCTDataset
from typing import List
from experiment import CovidCTClassification

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config")


def collect_dataloaders(config: ConfigDict) -> List[DataLoader]:
    loaders = []
    for set in ["train", "val", "test"]:
        data = CovidCTDataset(
            base_path=config.data.base_path,
            split=set,
            with_aug=config.data.with_aug,
            with_cgan=config.data.with_cgan,
            required_transform=config.data.required_transform,
            optional_transform=config.data.optional_transform,
        )
        loader = DataLoader(
            data,
            shuffle=config.training.shuffle_data,
            batch_size=config.training.batch_size,
            num_workers=config.training.num_workers,
            pin_memory=config.training.pin_memory,
        )
        loaders.append(loader)
    return loaders


def main(_):
    config = FLAGS.config

    # https://pytorch.org/docs/stable/notes/randomness.html
    if config.deterministic:
        seed_everything(config.manual_seed, workers=True)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    logger = TensorBoardLogger(save_dir=config.logging.save_dir, log_graph=True)

    logger.log_hyperparams(config)

    if config.model.from_torch:
        model = eval(
            "models.%s(pretrained=%r)" % (config.model.name, config.model.pretrained)
        )
    else:
        model = config.model.local_model()

    trainloader, valloader, testloader = collect_dataloaders(config)

    experiment = CovidCTClassification(model, config)

    callbacks = []
    if config.training.early_stopping.do:
        early_stopping = EarlyStopping(
            monitor=config.training.early_stopping.metric,
            patience=config.training.early_stopping.patience,
            min_delta=config.training.early_stopping.min_delta,
            mode=config.training.early_stopping.mode,
            check_finite=True,
            verbose=True,
        )
        callbacks.append(early_stopping)

    trainer = Trainer(
        logger=logger,
        accelerator="dp",
        callbacks=callbacks,
        max_epochs=config.training.max_epochs,
        gpus=config.training.gpus,
        log_every_n_steps=2,
        deterministic=config.deterministic,
        sync_batchnorm=True,
    )

    trainer.fit(
        model=experiment, train_dataloaders=trainloader, val_dataloaders=valloader
    )

    trainer.test(dataloaders=testloader)


if __name__ == "__main__":
    app.run(main)

