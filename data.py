import os
import glob

from argparse import ArgumentParser

from typing import Callable, Optional, Sequence

from torch.utils.data import DataLoader

import monai.data
import monai.utils
from monai.utils import first, set_determinism
from monai.transforms import (
    apply_transform, 
    AddChanneld,
    Compose, OneOf, 
    LoadImaged,
    Orientationd,
    RandFlipd, RandZoomd, RandScaleCropd,
    RandHistogramShiftd, 
    Resized,
    ScaleIntensityd,
    ScaleIntensityRanged, 
    ToTensord,
)
from monai.data import list_data_collate

from pytorch_lightning import LightningDataModule

class UnpairedDataset(monai.data.Dataset, monai.transforms.Randomizable):
    def __init__(
        self,
        keys: Sequence, 
        datasets: Sequence, 
        transform: Optional[Callable] = None,
        length: Optional[Callable] = None, 
        batch_size: int = 32
    ) -> None:
        self.keys = keys
        self.datasets = datasets
        self.length = length
        self.batch_size = batch_size
        self.transform = transform

    def __len__(self) -> int:
        if self.length is None:
            return min((len(dataset) for dataset in self.datasets))
        else: 
            return self.length

    def _transform(self, index: int):
        data = {}
        self.R.seed(index)
        for key, dataset in zip(self.keys, self.datasets):
            rand_idx = self.R.randint(0, len(dataset)) 
            data[key] = dataset[rand_idx]
        
        if self.transform is not None:
            data = apply_transform(self.transform, data)

        return data


# Region/Tissue Window  Level
# brain 80  40
# lungs 1500    -600
# liver 150 30
# Soft tissues  250 50
# bone  1800    400

def windowed(level, width):
   """
   Function to display an image slice
   Input is a numpy 2D array
   """
   maxval = level + width/2
   minval = level - width/2
   return minval, maxval

class CustomDataModule(LightningDataModule):
    def __init__(self, 
        image3d_dirs: str = "path/to/dir", 
        image2d_dirs: str = "path/to/dir", 
        shape: int = 256,
        batch_size: int = 32
    ):
        super().__init__()
        self.image3d_dirs = image3d_dirs, 
        self.image2d_dirs = image2d_dirs, 
        self.batch_size = batch_size
        self.shape = shape
        # self.setup()
        def glob_files(folders: str=None, extension: str='*.nii.gz'):
            assert folders is not None
            paths = [glob.glob(os.path.join(folder, extension)) for folder in folders]
            files = sorted([item for sublist in paths for item in sublist])
            print(len(files))
            print(files[:1])
            return files
            
        self.train_image3d_files = glob_files(folders=image3d_dirs, extension='*.nii.gz')
        self.train_image2d_files = glob_files(folders=image2d_dirs, extension='*.png')
        
        self.val_image3d_files = glob_files(folders=image3d_dirs, extension='*.nii.gz') # TODO
        self.val_image2d_files = glob_files(folders=image2d_dirs, extension='*.png')
        
        self.test_image3d_files = glob_files(folders=image3d_dirs, extension='*.nii.gz') # TODO
        self.test_image2d_files = glob_files(folders=image2d_dirs, extension='*.png')

    def setup(self, stage: Optional[str]=None):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        set_determinism(seed=0)

    def train_dataloader(self):
        self.train_transforms = Compose(
            [
                LoadImaged(keys=["image3d", "image2d"]),
                AddChanneld(keys=["image3d", "image2d"],),
                Orientationd(keys=["image3d"], axcodes="ARI"),
                RandFlipd(keys=["image3d"], prob=0.5, spatial_axis=0),
                # Spacingd(keys=["image3d"], pixdim=(1.0, 1.0, 0.25)),  
                # ScaleIntensityd(keys=["image2d", "image3d"], minv=0.0, maxv=1.0,),
                ScaleIntensityd(keys=["image2d"], minv=0.0, maxv=1.0,),
                OneOf([
                    ScaleIntensityd(keys=["image3d"], minv=0.0, maxv=1.0,),
                    ScaleIntensityRanged(keys=["image3d"], clip=True,  # LUNG
                        a_min=windowed(-600, 1500)[0], 
                        a_max=windowed(-600, 1500)[1],
                        b_min=0.0,
                        b_max=1.0),
                    ScaleIntensityRanged(keys=["image3d"], clip=True,  # SOFT TISSUE
                        a_min=windowed(50, 250)[0], 
                        a_max=windowed(50, 250)[1],
                        b_min=0.0,
                        b_max=1.0),
                    ScaleIntensityRanged(keys=["image3d"], clip=True,  # BONE
                        a_min=windowed(400, 1800)[0], 
                        a_max=windowed(400, 1800)[1],
                        b_min=0.0,
                        b_max=1.0),
                    ],
                ), 
                RandHistogramShiftd(keys=["image3d"], num_control_points=10, prob=0.8),
                # RandStdShiftIntensityd(keys=["image3d"], prob=0.2),
                # RandShiftIntensityd(keys=["image3d"], prob=0.2),
                # RandScaleIntensityd(keys=["image3d"], prob=0.2),
                # RandAdjustContrastd(keys=["image3d"], prob=0.8),
                RandZoomd(keys=["image3d"], prob=1.0, min_zoom=0.8, max_zoom=1.2), 
                RandZoomd(keys=["image2d"], prob=1.0, min_zoom=0.8, max_zoom=1.2), 
                RandScaleCropd(keys=["image3d"], 
                               roi_scale=(.75, .75, .75), 
                               max_roi_scale=(1.2, 1.2, 1.2), 
                               random_center=True, 
                               random_size=True),
                RandScaleCropd(keys=["image2d"], 
                               roi_scale=(0.75, 0.75), 
                               max_roi_scale=(1.2, 1.2), 
                               random_center=True, 
                               random_size=True),
                Resized(keys=["image3d"], spatial_size=(self.shape, self.shape, self.shape//1), mode=["area"],),
                Resized(keys=["image2d"], spatial_size=(self.shape, self.shape), mode=["area"],),
                # RandScaleCropd((256, 256, 256), random_size=False),
                ToTensord(keys=["image3d", "image2d"],),
            ]
        )

        self.train_datasets = UnpairedDataset(
            keys=["image3d", "image2d"],
            datasets=[self.train_image3d_files, self.train_image2d_files], 
            transform=self.train_transforms,
            length=1000,
            batch_size=self.batch_size,
        )


        # self.train_datasets = CacheDataset(
        #     data=self.train_files, 
        #     cache_rate=1.0, 
        #     num_workers=4,
        #     transform=self.train_transforms,
        # )

        self.train_loader = DataLoader(
            self.train_datasets, 
            batch_size=self.batch_size, 
            num_workers=4, 
            collate_fn=list_data_collate,
            shuffle=True,
        )
        return self.train_loader

    def val_dataloader(self):
        self.val_transforms = Compose(
            [
                LoadImaged(keys=["image3d", "image2d"]),
                AddChanneld(keys=["image3d", "image2d"],),
                Orientationd(keys=["image3d"], axcodes="ARI"),
                RandFlipd(keys=["image3d"], prob=0.5, spatial_axis=0),
                # Spacingd(keys=["image3d"], pixdim=(1.0, 1.0, 0.25)),  
                # ScaleIntensityd(keys=["image2d", "image3d"], minv=0.0, maxv=1.0,),
                ScaleIntensityd(keys=["image2d"], minv=0.0, maxv=1.0,),
                ScaleIntensityRanged(keys=["image3d"], clip=True, 
                    a_min=windowed(400, 1800)[0], 
                    a_max=windowed(400, 1800)[1],
                    b_min=0.0,
                    b_max=1.0),
                # ScaleIntensityd(keys=["image3d"], minv=0.0, maxv=1.0,),
                Resized(keys=["image3d"], spatial_size=(self.shape, self.shape, self.shape//1), mode=["area"],),
                Resized(keys=["image2d"], spatial_size=(self.shape, self.shape), mode=["area"],),
                # RandScaleCropd((256, 256, 256), random_size=False),
                ToTensord(keys=["image3d", "image2d"],),
            ]
        )

        self.val_datasets = UnpairedDataset(
            keys=["image3d", "image2d"],
            datasets=[self.val_image3d_files, self.val_image2d_files], 
            transform=self.val_transforms,
            length=200,
            batch_size=self.batch_size,
        )

        # self.val_datasets = CacheDataset(
        #     data=self.val_files, 
        #     cache_rate=1.0, 
        #     num_workers=4,
        #     transform=self.val_transforms,
        # )
        
        self.val_loader = DataLoader(
            self.val_datasets, 
            batch_size=self.batch_size, 
            num_workers=4, 
            collate_fn=list_data_collate,
            shuffle=False,
        )
        return self.val_loader


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--datadir", type=str, default='drive/MyDrive/Data/ChestXRLungSegmentation/', 
                        help="logging directory")

    hparams = parser.parse_args()

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
        batch_size = 1
    )

    datamodule.setup()

    debug_data = first(datamodule.val_dataloader())
    print(debug_data['image3d'].shape)
    print(debug_data['image2d'].shape)