import os 
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
import numpy as np

import torchvision

from typing import Optional, Tuple, Type

from monai.networks import normal_init
from monai.networks.nets import UNet, DenseNet121
from monai.networks.layers import Norm, Act

from pytorch3d.structures import Volumes
from pytorch3d.renderer.cameras import (
    CamerasBase,
    FoVOrthographicCameras,
    FoVPerspectiveCameras,
    OpenGLOrthographicCameras,
    OpenGLPerspectiveCameras,
    OrthographicCameras,
    PerspectiveCameras,
    SfMOrthographicCameras,
    SfMPerspectiveCameras,
    camera_position_from_spherical_angles,
    get_world_to_view_transform,
    look_at_rotation,
    look_at_view_transform,
)

from pytorch3d.renderer import (
    VolumeRenderer, 
    GridRaysampler, 
    NDCGridRaysampler, MonteCarloRaysampler, 
    EmissionAbsorptionRaymarcher, AbsorptionOnlyRaymarcher, 
)

from pytorch3d.transforms import (
    so3_exponential_map,
)

from pytorch_lightning import LightningModule

# Mapper
class CNNMapper(nn.Module):
    def __init__(self, 
                 input_dim: int = 1,
                 output_dim: int = 1,
    ): 
        super().__init__()
        self.vnet = nn.Sequential(
            UNet(
                dimensions=3,
                in_channels=input_dim,
                out_channels=output_dim+1, # value and alpha
                channels=(16, 32, 64, 128, 256,), #(20, 40, 80, 160, 320), #(32, 64, 128, 256, 512),
                strides=(2, 2, 2, 2),
                num_res_units=3,
                kernel_size=3,
                up_kernel_size=5,
                act=("LeakyReLU", {"negative_slope": 0.2, "inplace": True}),
                # norm=Norm.BATCH,
                dropout=0.5,
            ), 
            nn.Sigmoid()    
        )
        LAMBDA = 0.02
        self.alpha = nn.Parameter(LAMBDA*torch.ones(1)).requires_grad_(False)
        self.scaler = torch.abs(self.alpha).requires_grad_(False)

    def forward(self, raw_data: torch.Tensor, factor=None) -> torch.Tensor:
        B, C, D, H, W = raw_data.shape   
        self.scaler = torch.abs(self.alpha)
        scaler = self.scaler if factor is None else factor

        concat = self.vnet(raw_data)        
        values = concat[:,[0],:,:,:]
        alphas = concat[:,[1],:,:,:]

        # values = raw_data
        # alphas = self.vnet(raw_data)      

        features = torch.cat([values, alphas * scaler], dim=1)
        return features

class Viewer(nn.Module):
    def __init__(self,  
                 visualizer: VolumeRenderer = None, 
                 volumes: Volumes = None,
                 shape: int = 256,
    ) -> None:
        super().__init__()
        assert visualizer is not None and volumes is not None
        self.visualizer = visualizer
        self.volumes = volumes
        self.shape = shape 
        
    def forward(self, features: torch.Tensor, cameras: Type[CamerasBase]=None) -> torch.Tensor:
        assert cameras is not None
        B, C, D, H, W = features.shape    

        values = features[:,[0],:,:,:]
        alphas = features[:,[1],:,:,:]    
        # print(values.shape, alphas.shape)

        self.volumes = self.volumes.to(features.device)
        self.volumes._set_features(values.repeat(1,3,1,1,1))
        self.volumes._set_densities(alphas)
        
        screen_RGBA, _ = self.visualizer(cameras=cameras, volumes=self.volumes) #[...,:3]
        screen_RGBA = screen_RGBA.reshape(B, self.shape, self.shape, 4).permute(0,3,2,1) # 3 for NeRF

        value = screen_RGBA[:,:3].mean(dim=1, keepdim=True)
        value = value - value.min()
        value = value / value.max()
        return value

def RandomCameras(
        dist: float = 3.0, 
        elev: float = 0.0, 
        azim: float = 0.0, 
        batch_size: int = 32,
        random: bool = False, 
    ): 
    # R0, T = look_at_view_transform(dist, elev, azim)
    if random:
        rand = 180*np.random.randint(0, 2)
        elev = np.random.uniform( -5,  5) + 180*rand
        azim = np.random.uniform( -5,  5) + 180*rand

    #     R0, T = look_at_view_transform(dist, elev, azim)
    #     sin = np.sin(azim/180*np.pi)
    #     cos = np.cos(azim/180*np.pi)
    #     Rt = torch.from_numpy(np.eye(3)).float()
    #     if np.random.randint(0, 3)==0:
    #         Rt = torch.tensor([[1,0,0],
    #                            [0,cos,-sin],
    #                            [0,sin,cos]]).type(torch.float)
                
    #     elif np.random.randint(0, 3)==1:
    #         Rt = torch.tensor([[cos,0,cos],
    #                            [0,1,0],
    #                         [-sin,0,cos]]).type(torch.float)
    #     elif np.random.randint(0, 3)==2:      
    #         Rt = torch.tensor([[cos,-sin,0],
    #                    [sin,cos,0],
    #                    [0,0,1]]).type(torch.float)  
    #     R = torch.matmul(R0, Rt)
    # else:
    #     R = R0
    R, T = look_at_view_transform(dist, elev, azim)
    R = R.repeat(batch_size, 1, 1)
    T = T.repeat(batch_size, 1)

    znear = 0.1 * torch.ones(batch_size) if random else \
            0.1 * torch.ones(batch_size) # 0.5*torch.ones(batch_size) * 10 + 0.1
    zfar = 4.0 * torch.ones(batch_size) if random else \
           4.0 * torch.ones(batch_size) # 0.5*torch.ones(batch_size) * 4 + 1 + znear
    fov = torch.ones(batch_size) * 60 + (torch.randn(batch_size)) * 5 if random else \
          torch.ones(batch_size) * 60 + 0
    aspect_ratio = 1.3 * torch.ones(batch_size) if random else \
                   1.3 * torch.ones(batch_size)
                #    torch.rand(batch_size) * 0.5 + 0.5
                #    1.0 * torch.rand(batch_size) + 1.2 if random else \
                   
    return FoVPerspectiveCameras(R=R, T=T, znear=znear, zfar=zfar, fov=fov, aspect_ratio=aspect_ratio)

#Custom Lightning module
class CustomLightningModule(LightningModule):
    def __init__(self, hparams, **kwargs):
        super().__init__()
        self.logdir = hparams.logdir
        self.lr = hparams.lr
        self.b1 = hparams.b1
        self.b2 = hparams.b2
        self.shape = hparams.shape
        self.batch_size = hparams.batch_size
        
        self.save_hyperparameters()

        # Deterministic Cameras for validation
        self.detcams = RandomCameras(batch_size=self.batch_size, random=False)
        self.varcams = RandomCameras(batch_size=self.batch_size, random=True)

        self.mapper = CNNMapper(
            input_dim = 1,
            output_dim = 1,
        )

        self.raysampler = NDCGridRaysampler(
            image_width = self.shape,
            image_height = self.shape,
            n_pts_per_ray = self.shape * 2,
            min_depth = 0.001,
            max_depth = 4.0,
        )

        self.raymarcher = EmissionAbsorptionRaymarcher()

        self.visualizer = VolumeRenderer(
            raysampler = self.raysampler, 
            raymarcher = self.raymarcher,
        )
        
        print("Self Device: ", self.device)
        self.features = torch.randn((self.batch_size, 3, self.shape, self.shape, self.shape))
        self.densities = torch.randn((self.batch_size, 1, self.shape, self.shape, self.shape)) 
        self.volumes = Volumes(
            features = self.features,
            densities = self.densities,
            voxel_size = 4.0 / self.shape,
        ) 

        self.viewer = Viewer(
            visualizer = self.visualizer, 
            volumes = self.volumes,
            shape = self.shape
        ) 

        self.gen = nn.Sequential(
            self.mapper,
            self.viewer,
        )

        self.discrim = nn.Sequential(
            UNet(
                dimensions=2,
                in_channels=1,
                out_channels=1, # value and alpha
                channels=(32, 64, 128, 256, 512),
                strides=(2, 2, 2, 2),
                num_res_units=2,
                kernel_size=3,
                up_kernel_size=5,
                act=("LeakyReLU", {"negative_slope": 0.2, "inplace": True}),
                # norm=Norm.BATCH,
                # dropout=0.5,
            ), 
            # nn.Sigmoid()  
        )

        # self.discrim = nn.Sequential(
        #     DenseNet121(
        #         spatial_dims=2, 
        #         in_channels=1, 
        #         out_channels=1,
        #         pretrained=False, 
        #         # dropout_prob=0.5
        #     ),
        #     # nn.Sigmoid(),
        # )

        # self.gen.apply(normal_init)
        # self.discrim.apply(normal_init)

    def discrim_step(self, fake_images: torch.Tensor, real_images: torch.Tensor):
        real_logits = self.discrim(real_images)
        fake_logits = self.discrim(fake_images) #.detach()
        real_loss = F.softplus(-real_logits)
        fake_loss = F.softplus(fake_logits)
        return real_loss.mean() + fake_loss.mean() 

    def gen_step(self, fake_images: torch.Tensor, real_images: torch.Tensor):
        fake_logits = self.discrim(fake_images)
        return F.softplus(-fake_logits).mean()
        
    def forward(self, image3d: torch.Tensor, cameras: Type[CamerasBase]=None, factor: float=None):
        if cameras is None:
            cameras = self.detcams
        cameras = cameras.to(image3d.device)
        mapped_data = (self.gen[0].forward(image3d, factor=factor))
        # Padding mapped
        viewed_data = (self.gen[1].forward(mapped_data, cameras=cameras))
        return viewed_data, mapped_data

    def training_step(self, batch, batch_idx, optimizer_idx, stage: Optional[str]='train'):
        image3d, image2d = batch["image3d"], batch["image2d"]
        # generate images
        with torch.no_grad():
            self.varcams = RandomCameras(batch_size=self.batch_size, random=True)
        viewed_, mapped_ = self.forward(image3d, self.varcams, factor=0.02)

        # Log to tensorboard, change to callback later
        if batch_idx==0:
            with torch.no_grad():
                scaled = viewed_.clone()
                for i in range(self.batch_size):
                    scaled[i] -= torch.min(scaled[i])
                    scaled[i] /= torch.max(scaled[i]) + 1e-6
                viz = torch.cat([image3d[...,64], mapped_[:,:1,...,64], 
                                 mapped_[:,[1],...,64] / (self.gen[0].scaler), 
                                 viewed_, image2d], dim=-1)
                grid = torchvision.utils.make_grid(viz, normalize=False, scale_each=False, nrow=1, padding=0)
                tensorboard = self.logger[0].experiment
                tensorboard.add_image(f'{stage}_samples', grid, self.current_epoch)
                # for name, hist in self.gen[0].named_parameters():
                #     tensorboard.add_histogram(f'{stage}_{name}', hist, self.current_epoch)

        # train generator
        if optimizer_idx == 0:
            g_loss = self.gen_step(fake_images=viewed_, real_images=image2d)
            self.log(f'{stage}_g_loss', g_loss, on_step=True, prog_bar=True, logger=True)
            self.log(f'{stage}_scaler', (self.gen[0].scaler.detach()), on_step=True, prog_bar=True, logger=True)
            # return {'loss': g_loss}
            r_loss = 2e+1*nn.L1Loss()(mapped_[:,[0]], image3d) \
                   + 2e-1*nn.L1Loss()(mapped_[:,[1]] / (self.gen[0].scaler + 1e-8), torch.rand_like(mapped_[:,[1]]))

            self.log(f'{stage}_r_loss', r_loss, on_step=True, prog_bar=True, logger=True)
            return {'loss': g_loss+r_loss}  

        # train discriminator
        elif optimizer_idx == 1:
            d_loss = self.discrim_step(fake_images=viewed_.detach(), real_images=image2d)
            self.log(f'{stage}_d_loss', d_loss, on_step=True, prog_bar=True, logger=True)
            return {'loss': d_loss}



    def evaluation_step(self, batch, batch_idx, stage: Optional[str]='evaluation'):   
        image3d, image2d = batch["image3d"], batch["image2d"]     
        with torch.no_grad():
            self.detcams = RandomCameras(batch_size=self.batch_size, random=False)
        viewed_, mapped_ = self.forward(image3d, self.detcams, factor=0.02)
        
        # Log to tensorboard, change to callback later
        if batch_idx==0:
            with torch.no_grad():
                scaled = viewed_.clone()
                for i in range(self.batch_size):
                    scaled[i] -= torch.min(scaled[i])
                    scaled[i] /= torch.max(scaled[i]) + 1e-6
                viz = torch.cat([image3d[...,64], mapped_[:,:1,...,64], 
                                 mapped_[:,[1],...,64] / (self.gen[0].scaler), 
                                 viewed_, image2d], dim=-1)
                grid = torchvision.utils.make_grid(viz, normalize=False, scale_each=False, nrow=1, padding=0)
                tensorboard = self.logger[0].experiment
                tensorboard.add_image(f'{stage}_samples', grid,  self.current_epoch)
                torchvision.utils.save_image(grid, 
                    os.path.join(self.logdir, f'{stage}_samples_{self.current_epoch}.png'))
                # for name, hist in self.gen[0].named_parameters():
                #     tensorboard.add_histogram(f'{stage}_{name}', hist, self.current_epoch)

        g_loss = self.gen_step(fake_images=viewed_, real_images=image2d)
        d_loss = self.discrim_step(fake_images=viewed_, real_images=image2d)
        # return {"g_loss": g_loss, "d_loss": d_loss}
        r_loss = 2e+1*nn.L1Loss()(mapped_[:,[0]], image3d) \
               + 2e-1*nn.L1Loss()(mapped_[:,[1]] / (self.gen[0].scaler + 1e-8), torch.rand_like(mapped_[:,[1]]))

        self.log(f'{stage}_r_loss', r_loss, on_step=True, prog_bar=True, logger=True)
            
        return {"g_loss": g_loss, "r_loss": r_loss, "d_loss": d_loss}

    def validation_step(self, batch, batch_idx):
        return self.evaluation_step(batch, batch_idx, stage='validation')

    def test_step(self, batch, batch_idx):
        return self.evaluation_step(batch, batch_idx, stage='test')

    def evaluation_epoch_end(self, outputs, stage: Optional[str]='evaluation'):
        g_loss = torch.stack([x["g_loss"] for x in outputs]).mean()
        d_loss = torch.stack([x["d_loss"] for x in outputs]).mean()
        
        self.log(f'{stage}_g_loss', g_loss, on_step=False, prog_bar=True, logger=True)
        self.log(f'{stage}_d_loss', d_loss, on_step=False, prog_bar=True, logger=True)

        r_loss = torch.stack([x["r_loss"] for x in outputs]).mean()
        self.log(f'{stage}_r_loss', r_loss, on_step=False, prog_bar=True, logger=True)
        self.log(f'{stage}_scaler', (self.gen[0].scaler.detach()), on_step=False, prog_bar=True, logger=True)

    def validation_epoch_end(self, outputs):
        return self.evaluation_epoch_end(outputs, stage='validation')
    
    def test_epoch_end(self, outputs):
        return self.evaluation_epoch_end(outputs, stage='test')

    def configure_optimizers(self):
        b1 = self.b1
        b2 = self.b2

        opt_g = torch.optim.AdamW(self.gen.parameters(), lr=1*(self.lr or self.learning_rate), betas=(b1, b2))
        opt_d = torch.optim.AdamW(self.discrim.parameters(), lr=1*(self.lr or self.learning_rate), betas=(b1, b2))
        return (
            {'optimizer': opt_g, 'frequency': 2},
            {'optimizer': opt_d, 'frequency': 1}
        )