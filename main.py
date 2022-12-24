import pytorch_lightning as pl
from dataset import MyDataModule
from train import MyModel
import os
import warnings
from pytorch_lightning.loggers import WandbLogger

warnings.filterwarnings("ignore")
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
pl.seed_everything(1234)
dm = MyDataModule('/mnt/epnfs/zhshen/DE_code_0904/UIEB',
                        batch_size = 32,
                        num_workers = 64)
n_epochs = 100
samples_dir = './samples'
model = MyModel(samples_dir)
logger = WandbLogger(name='unet_test', project='Models_test')
# model.load_state_dict(torch.load(os.path.join('/mnt/epnfs/zhshen/DE_code_0904/results/mynet_uieb', 'model.pt')))
trainer = pl.Trainer(
    accelerator="auto",
    devices=[5],  # limiting got iPython runs
    max_epochs=n_epochs,
    check_val_every_n_epoch=10,
    auto_lr_find=True,
    logger = logger
)
# trainer.tune(model, dm)
trainer.fit(model, dm)
torch.save(model.state_dict(), os.path.join(samples_dir,'model.pt'))