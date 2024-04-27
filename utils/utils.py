import torch
import pytorch_lightning as pl
from torchvision.models import resnet18
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, MultiStepLR
from .models.resnet import ResNet18

class PrintCallback(pl.Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        print(f"Training loss: {trainer.callback_metrics['train_loss']}, Accuracy: {trainer.callback_metrics['train_accuracy']}")
        
    def on_validation_end(self, trainer, pl_module):
        print(f"Validation loss: {trainer.callback_metrics['val_loss']}, Accuracy: {trainer.callback_metrics['val_accuracy']}")

def poison_labels(dataset):
    dataset.targets = torch.randint(0, 10, (len(dataset),))

def resolve_optimizer(cfg, params):
    if cfg["optimizer"] == "sgd":
        return SGD(
            params,
            lr=cfg["optimizer_lr"],
            momentum=cfg["optimizer_momentum"],
            weight_decay=cfg["optimizer_weight_decay"], 
            nesterov=True           
        )
    elif cfg["optimizer"] == "adam":
        return Adam(
            params,
            lr=cfg["optimizer_lr"],
            weight_decay=cfg["optimizer_weight_decay"],
        )
    else:
        raise NotImplementedError


def resolve_lr_scheduler(cfg, optimizer):
    if cfg["lr_scheduler"] == "poly":
        return LambdaLR(
            optimizer,
            lambda ep: max(1e-6, (1 - ep / cfg["epochs"]) ** cfg["lr_scheduler_power"])
        )
    elif cfg["lr_scheduler"] == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=cfg["epochs"]
        )
    elif cfg["lr_scheduler"] == "multistep":
        return MultiStepLR(
            optimizer,
            milestones=cfg["milestones"]
        )
    elif cfg["lr_scheduler"] == "none":
        return None
    else:
        raise NotImplementedError
    
def resolve_model(dataset):
    return {
        "cifar10": ResNet18(),
        "utkface": resnet18(),
        "fgnet": resnet18(), 
        "morph": resnet18(weights='IMAGENET1K_V1'),
        "agedb": resnet18()
    }[dataset]

def resolve_classes(dataset):
    return {
        "cifar10": 10,
        "utkface": 10,
        "fgnet": 7, 
        "morph": 8,
        "agedb": 10
    }[dataset]

def resolve_loss(dataset):
    w = torch.tensor(0)
    if dataset == "utkface":
        w = torch.tensor([3062, 1531, 7344, 4537, 2245, 2299, 1318, 699, 504, 137]).float()
        w = 23676. / w
    elif dataset == "fgnet":
        w = torch.tensor([332, 314, 129, 68, 38, 14, 6]).float()
        w = 900. / w
    elif dataset == "morph":
        w = torch.tensor([torch.inf, 5094, 11852, 11286, 8812, 2716, 240, 12]).float()
        #w = torch.tensor([16946, 20098, 2968]).float()
        w = 40000. / w
    elif dataset == "agedb":
        w = torch.tensor([42, 449, 2251, 3257, 2872, 2357, 1818, 1247, 428, 63]).float()
        w = 14784. / w

    if torch.cuda.is_available():
        w = w.to("cuda")
    return {
        "cifar10": torch.nn.CrossEntropyLoss(),
        "utkface": torch.nn.CrossEntropyLoss(weight=w),
        "fgnet": torch.nn.CrossEntropyLoss(weight=w),
        "morph": torch.nn.CrossEntropyLoss(weight=w),
        "agedb": torch.nn.CrossEntropyLoss(weight=w)
    }[dataset]