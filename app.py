import os

from argparse import ArgumentParser

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor

from callback import *
from model import *
from data import *

if __name__ == "__main__":
    parser = ArgumentParser()
    # System arguments: --gpus is default argument for cli
    parser.add_argument("--gpus", type=int, default=0, help="number of GPUs")
    parser.add_argument("--conda_env", type=str, default="NeRP")
    parser.add_argument("--notification_email", type=str, default="quantm88@gmail.com")
    
    # Model arguments
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--shape", type=int, default=256, help="spatial size of the tensor")
    parser.add_argument("--epochs", type=int, default=501, help="number of epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: 1st order momentum")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: 2nd order momentum")
    parser.add_argument("--ckpt", type=str, default=None, help="path to checkpoint")
    parser.add_argument("--logdir", type=str, default='logs', help="logging directory")
    parser.add_argument("--datadir", type=str, default='drive/MyDrive/Data/ChestXRLungSegmentation/', 
                        help="logging directory")

    # parser = Trainer.add_argparse_args(parser)
    
    # Collect the hyper parameters
    hparams = parser.parse_args()

    # Seed the application
    set_determinism(2023)

    # Callback
    checkpoint_callback = ModelCheckpoint(
        # monitor='validation_r_loss',
        dirpath=hparams.logdir,
        filename='{epoch:02d}-{validation_g_loss:.2f}-{validation_d_loss:.2f}',
        save_top_k=-1,
        save_last=True,
        # mode='min',
        every_n_epochs=10, 
    )
    lr_callback = LearningRateMonitor(logging_interval='step')
    tensorboard_callback = TensorboardGenerativeModelImageProjector()
    # Logger
    tensorboard_logger = TensorBoardLogger(save_dir=hparams.logdir, log_graph=True)

    # Init model with callbacks
    trainer = Trainer.from_argparse_args(
        hparams, 
        max_epochs=hparams.epochs,
        resume_from_checkpoint = hparams.ckpt, #"logs/default/version_0/epoch=50.ckpt",
        logger=[tensorboard_logger],
        callbacks=[
            lr_callback,
            checkpoint_callback, 
            # tensorboard_callback
        ],
        # precision=16,
        # stochastic_weight_avg=True,
        auto_scale_batch_size=True, 
        # gradient_clip_val=2.0, 
        # gradient_clip_algorithm='norm', #'norm', #'value'
    )

    # Create data module
    train_image3d_dirs = [os.path.join(hparams.datadir, 'NSCLC/processed/images'),]
    train_label3d_dirs = [os.path.join(hparams.datadir, 'NSCLC/processed/labels'),]

    train_image2d_dirs = [
        os.path.join(hparams.datadir, 'JSRT/processed/images/'), 
        os.path.join(hparams.datadir, 'ChinaSet/processed/images/'), 
        os.path.join(hparams.datadir, 'Montgomery/processed/images/'),
    ]
    train_label2d_dirs = [
        os.path.join(hparams.datadir, 'JSRT/processed/labels/'), 
        os.path.join(hparams.datadir, 'ChinaSet/processed/labels/'), 
        os.path.join(hparams.datadir, 'Montgomery/processed/labels/'),
    ]

    val_image3d_dirs = train_image3d_dirs
    val_image2d_dirs = train_image2d_dirs

    test_image3d_dirs = train_image3d_dirs
    test_image2d_dirs = train_image2d_dirs

    datamodule = CustomDataModule(
        image3d_dirs = train_image3d_dirs, 
        image2d_dirs = train_image2d_dirs, 
        batch_size = hparams.batch_size, 
        shape = hparams.shape
    )
    datamodule.setup()

    model = CustomLightningModule(
        hparams = hparams
    )

    trainer.fit(model, datamodule)

    # test

    # serve
