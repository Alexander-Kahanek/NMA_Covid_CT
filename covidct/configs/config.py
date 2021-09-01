import ml_collections
from ml_collections import config_dict
from torch.optim import SGD, Adam
import torch.nn as nn, torch
from torchvision.transforms import Compose, Resize, Normalize, ToTensor


def get_config():
    config = ml_collections.ConfigDict()

    # determinism and reproducibility
    config.deterministic = True
    config.manual_seed = 100

    # model instance
    config.model = ml_collections.ConfigDict()
    config.model.img_size = (256, 256)
    config.model.from_torch = True
    config.model.name = "resnet50"
    config.model.pretrained = True
    config.model.freeze_weights = False
    config.model.local_model = config_dict.placeholder(nn.Module)
    config.model.num_classes = 1

    # optimizer params
    config.optimizer = ml_collections.ConfigDict()
    config.optimizer.optim = Adam
    config.optimizer.params = ml_collections.ConfigDict()
    config.optimizer.params.lr = 0.001
    config.optimizer.params.weight_decay = 0.0001
    if config.optimizer.optim == SGD:
        config.optimizer.params.momentum = 0.9
    elif config.optimizer.optim == Adam:
        config.optimizer.betas = (0.9, 0.99)
    config.optimizer.scheduler = config_dict.placeholder(object)

    # training params
    config.training = ml_collections.ConfigDict()
    config.training.batch_size = 32
    config.training.max_epochs = 100
    config.training.num_workers = 20
    config.training.pin_memory = True
    config.training.shuffle_data = False
    config.training.early_stopping = ml_collections.ConfigDict()
    config.training.early_stopping.do = False
    config.training.early_stopping.metric = "val_acc"
    config.training.early_stopping.mode = "max"
    config.training.early_stopping.min_delta = 0.0
    config.training.early_stopping.patience = 5
    num_gpus = torch.cuda.device_count()
    config.training.gpus = None if num_gpus == 0 else num_gpus

    # dataset details
    config.data = ml_collections.ConfigDict()
    config.data.base_path = "/home/luke/work/covidct/data/"
    config.data.with_aug = True
    config.data.with_cgan = False
    if config.model.pretrained:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    else:
        mean = (0.5972, 0.5969, 0.5966)
        std = (0.3207, 0.3207, 0.3207)
    config.data.required_transform = Compose(
        [ToTensor(), Resize(config.model.img_size), Normalize(mean, std),]
    )
    config.data.optional_transform = config_dict.placeholder(Compose)

    # logging details
    config.logging = ml_collections.ConfigDict()
    config.logging.deterministic = True
    config.logging.debug = True
    config.logging.save_dir = "/home/luke/work/covidct/logs"

    return config
