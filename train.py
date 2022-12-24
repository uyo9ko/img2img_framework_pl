import torch
import torch.nn as nn
import pytorch_lightning as pl
import timm
import segmentation_models_pytorch as smp
import torchvision.utils as vutils
from loss import *
import os

class MyModel(pl.LightningModule):
    def __init__(self, samples_dir):
        super().__init__()
        self.loss_fn = VGGPerceptualLoss()
        self.model = smp.Unet(
            encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=3,                      # model output channels (number of classes in your dataset)
        )
        self.samples_dir = samples_dir
        self.lr = 1e-3

    def forward(self, x):
        return self.model(x)


    def training_step(self, batch, batch_idx):
        x, y, _= batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('loss',loss)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, y, names = batch
        y_hat = self(x)
        [vutils.save_image(y_hat[i], os.path.join(self.samples_dir, names[i])) for i in range(len(names))]
        loss = self.loss_fn(y_hat, y)
        return {'val_loss': loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.99, 0.999), eps=1e-08, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "loss"}
