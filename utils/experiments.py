import torch
import pytorch_lightning as pl
from sklearn.metrics import mean_squared_error

class Experiment(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.optimizer = cfg["optimizer"]
        self.model = cfg["model"]
        self.loss = cfg["loss"]

    def forward(self, inputs):
        out = self.model(inputs)
        return out
    
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        out = self.forward(x)        
        loss = self.loss(out.squeeze(), y.squeeze())        
        self.log("train_loss", loss)
        mse = mean_squared_error(y.cpu().numpy(), out.cpu().detach().numpy().argmax(axis=-1))
        self.log("train_mse", mse, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        out = self.forward(x)
        loss = self.loss(out.squeeze(), y.squeeze())
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        mse = mean_squared_error(y.cpu().numpy(), out.cpu().numpy().argmax(axis=-1))
        self.log("val_mse", mse, on_step=False, on_epoch=True)
        
        
    def configure_optimizers(self):
        optimizer = self.optimizer