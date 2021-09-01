from typing import Tuple
import pytorch_lightning as pl, torch, torch.nn as nn
from torch import Tensor
from ml_collections import ConfigDict
from torchmetrics.classification import Accuracy, F1, Precision, Specificity


class CovidCTClassification(pl.LightningModule):
    def __init__(self, model: nn.Module, config: ConfigDict) -> pl.LightningModule:
        super().__init__()
        self.example_input_array = torch.empty((1, 3, *config.model.img_size))
        self.model = self.config_model(model, config.model)
        self.optim_params = config.optimizer.params
        self.optim = config.optimizer.optim
        self.loss_fn = nn.BCEWithLogitsLoss(reduction="mean")
        self.accuracy_fn = Accuracy(
            num_classes=config.model.num_classes, dist_sync_on_step=True
        )
        self.f1_fn = F1(num_classes=config.model.num_classes, dist_sync_on_step=True)
        self.precision_fn = Precision(
            num_classes=config.model.num_classes, dist_sync_on_step=True
        )
        self.specificity_fn = Specificity(
            num_classes=config.model.num_classes, dist_sync_on_step=True
        )

    def config_model(self, model: nn.Module, model_config: ConfigDict) -> nn.Module:
        assert model_config.name in [
            "resnet50",
            "resnet101",
            "alexnet",
            "vgg16_bn",
            "googlenet",
            "vgg19_bn",
        ], "provided model not supported!"

        if not model_config.pretrained or model_config.local_model:
            return model

        if model_config.freeze_weights:
            for param in model.parameters():
                param.requires_grad = False

        model.eval()
        with torch.no_grad():
            tmp_out = model(self.example_input_array)
            n_feat = tmp_out.size(-1)

        classifier = nn.Linear(n_feat, model_config.num_classes)

        model = nn.Sequential(model, classifier)
        return model

    def forward(self, x: Tensor) -> Tensor:
        return self.model.forward(x)

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        x, y = batch
        y_hat = self(x).squeeze(1)
        loss = self.loss_fn(y_hat, y.float())
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def _shared_eval_step_end(
        self, outputs: dict
    ) -> Tuple[float, float, float, float, float]:
        y = outputs["y"]
        y_hat = outputs["y_hat"]
        loss = self.loss_fn(y_hat, y.float())
        acc = self.accuracy_fn(y_hat, y)
        f1 = self.f1_fn(y_hat, y)
        specificity = self.specificity_fn(y_hat, y)
        precision = self.precision_fn(y_hat, y)
        return loss, acc, f1, specificity, precision

    def validation_step(self, batch: Tensor, batch_idx: int) -> None:
        x, y = batch
        y_hat = self.model(x).squeeze(1)
        return {"y": y, "y_hat": y_hat}

    def validation_step_end(self, outputs: dict) -> None:
        loss, acc, f1_score, spec_score, prec_score = self._shared_eval_step_end(
            outputs
        )
        metrics = {
            "val_acc": acc,
            "val_loss": loss,
            "val_f1": f1_score,
            "val_specificity": spec_score,
            "val_precision": prec_score,
        }
        self.log_dict(metrics, logger=True, prog_bar=True)

    def test_step(self, batch: Tensor, batch_idx: int) -> None:
        x, y = batch
        y_hat = self.model(x).squeeze(1)
        return {"y": y, "y_hat": y_hat}

    def test_step_end(self, outputs: dict) -> None:
        loss, acc, f1_score, spec_score, prec_score = self._shared_eval_step_end(
            outputs
        )
        metrics = {
            "test_acc": acc,
            "test_loss": loss,
            "test_f1": f1_score,
            "test_specificity": spec_score,
            "test_precision": prec_score,
        }
        self.log_dict(metrics, logger=True, prog_bar=True)

    def configure_optimizers(self) -> None:
        optimizer = self.optim(self.model.parameters(), **self.optim_params)
        return {"optimizer": optimizer}
