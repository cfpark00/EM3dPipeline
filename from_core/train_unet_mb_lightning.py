import os
import comet_ml
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.callbacks import LearningRateMonitor

from model import LightVDM, UNetThin2
from dataset import get_dataset_3D_128_LH_CV

torch.set_float32_matmul_precision("medium")

def train(model,datamodule,):
    comet_logger = CometLogger(
        api_key=os.environ.get("COMET_API_KEY"),
        save_dir="../data/comet_logs",
        project_name="test_mb",
        experiment_name="test1",
    )

    trainer = Trainer(
        logger=comet_logger,
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        max_epochs=5000,
        gradient_clip_val=0.5,
        callbacks=[LearningRateMonitor()],

    )
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    # Ensure reproducibility
    seed_everything(7,workers=True)

    #settings
    batch_size = 16
    lr=1e-2
    num_workers = 8
    cropsize = 128

    n_steps=50000
    save_top=3
    pretrain_path=None


    #train
    datamodule = get_dataset
    model=EM2MB_UNet()
    train(model=model, datamodule=datamodule,deterministic=True)
