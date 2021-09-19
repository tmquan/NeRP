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
from monai.networks.nets import *
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


LAMBDA = 0.02

# Mapper
class LookUpTableMapper(nn.Module):
    def __init__(self, 
                 input_dim: int = 4096,
                 output_dim: int = 1,
                #  device: Union[str, torch.device] = 'cpu'
    ) -> None:
        super().__init__()
        assert output_dim==1 or output_dim==3
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.emb_value = nn.Embedding(num_embeddings=self.input_dim, embedding_dim=self.output_dim).requires_grad_(True)
        self.emb_alpha = nn.Embedding(num_embeddings=self.input_dim, embedding_dim=self.output_dim).requires_grad_(True)
        # self.alpha = nn.Parameter(torch.empty(1))
        # nn.init.constant_(self.alpha, 0.05).requires_grad_(True)
        # nn.init.normal_(self.emb_value.weight, 0.5, 0.5).requires_grad_(True)
        # nn.init.normal_(self.emb_alpha.weight, 0.5, 0.5).requires_grad_(True)
        self.emb_value.weight.data = \
            torch.from_numpy(np.concatenate([np.arange(self.input_dim)[:, np.newaxis] / self.input_dim]*self.output_dim, 
                                            axis=-1)).float()
        self.emb_alpha.weight.data.copy_(torch.ones(self.input_dim, self.output_dim) / 50 )
        

        self.alpha = nn.Parameter(LAMBDA*torch.ones(1)).requires_grad_(False)
        self.scaler = torch.abs(self.alpha).requires_grad_(False)

    def forward(self, raw_data: torch.Tensor, factor: float) -> torch.Tensor:
        B, C, D, H, W = raw_data.shape     
        
        self.scaler = torch.abs(self.alpha)
        scaler = self.scaler if factor is None else factor

        raw_data_long = (raw_data*(self.input_dim-1)).long()
        values = self.emb_value( raw_data_long )
        alphas = self.emb_alpha( raw_data_long ) 

        values = torch.clamp(values, 0, 1)
        alphas = torch.clamp(alphas, 0, 1)
        if self.output_dim == 1:
            # Convert luminance to RGB
            values = torch.reshape(values, [B,D,H,W,self.output_dim]).permute(0,4,1,2,3).repeat(1,3,1,1,1)
        elif self.output_dim == 3:
            values = torch.reshape(values, [B,D,H,W,self.output_dim]).permute(0,4,1,2,3).mean(dim=1, keepdim=True).repeat(1,3,1,1,1)
        alphas = torch.reshape(alphas, [B,D,H,W,1]).permute(0,4,1,2,3)

        # values = torch.relu(values)
        # alphas = torch.relu(alphas)
        
        features = torch.cat([values, alphas * scaler], dim=1)
        return features

def replace_conv2d(module, name):
    # go through all attributes of module nn.module (e.g. network or layer) and put batch norms if present
    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)
        if type(target_attr) == torch.nn.Conv2d:
            print('replaced: ', attr_str)
            CustomConv2d = EqualConv2d(target_attr.in_channels, 
                                        target_attr.out_channels, 
                                        target_attr.kernel_size,
                                        target_attr.stride, 
                                        target_attr.padding, 
                                        target_attr.dilation)
            setattr(module, attr_str, CustomConv2d)

    # iterate through immediate child modules. Note, the recursion is done by our code no need to use named_modules()
    for name, immediate_child_module in module.named_children():
        replace_conv2d(immediate_child_module, name)

def replace_conv3d(module, name):
    # go through all attributes of module nn.module (e.g. network or layer) and put batch norms if present
    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)
        if type(target_attr) == torch.nn.Conv3d:
            print('replaced: ', attr_str)
            CustomConv3d = EqualConv3d(target_attr.in_channels, 
                                        target_attr.out_channels, 
                                        target_attr.kernel_size,
                                        target_attr.stride, 
                                        target_attr.padding, 
                                        target_attr.dilation)
            setattr(module, attr_str, CustomConv3d)

    # iterate through immediate child modules. Note, the recursion is done by our code no need to use named_modules()
    for name, immediate_child_module in module.named_children():
        replace_conv3d(immediate_child_module, name)

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
                act=("PReLU"),
                norm=Norm.BATCH,
                dropout=0.5,
            ), 
            # UNet(
            #     num_classes=1,
            #     input_channels=1,
            #     num_layers=4,
            #     features_start=16,
            #     bilinear=False,
            # ),
            nn.Sigmoid()    
        )

        replace_conv3d(self.vnet, "self.vnet")
        # print(self.vnet)
        
        # LAMBDA = 0.02
        self.alpha = nn.Parameter(LAMBDA*torch.ones(1)).requires_grad_(False)
        self.scaler = torch.abs(self.alpha).requires_grad_(False)

    def forward(self, raw_data: torch.Tensor, factor=None, weight=None) -> torch.Tensor:
        B, C, D, H, W = raw_data.shape   
        self.scaler = torch.abs(self.alpha)
        scaler = self.scaler if factor is None else factor

        concat = self.vnet(raw_data)   
        if weight is None:
            weight = torch.rand([1], device=raw_data.device)     

        values = concat[:,[0],:,:,:]*(1-weight) + raw_data*(weight) # Randomly inject residuality
        alphas = concat[:,[1],:,:,:]

        # values = raw_data
        # alphas = self.vnet(raw_data)    

        # values = self.vnet(raw_data)     
        # alphas = torch.ones_like(values)

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
        
    def forward(self, 
        image3d: torch.Tensor, 
        features: torch.Tensor, 
        cameras: Type[CamerasBase]=None
    ) -> torch.Tensor:
        assert cameras is not None
        B, C, D, H, W = features.shape    

        # values = image3d
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

def total_variation(img: torch.Tensor) -> torch.Tensor:
    r"""Function that computes Total Variation according to [1].

    Args:
        img: the input image with shape :math:`(N, C, H, W)` or :math:`(C, H, W)`.

    Return:
         a scalar with the computer loss.

    Examples:
        >>> total_variation(torch.ones(3, 4, 4))
        tensor(0.)

    .. note::
       See a working example `here <https://kornia-tutorials.readthedocs.io/en/latest/
       total_variation_denoising.html>`__.

    Reference:
        [1] https://en.wikipedia.org/wiki/Total_variation
    """
    if not isinstance(img, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(img)}")

    if len(img.shape) < 3 or len(img.shape) > 5:
        raise ValueError(f"Expected input tensor to be of ndim 3 or 5, but got {len(img.shape)}.")
    if len(img.shape) == 4:
        pixel_dif1 = img[..., 1:, :] - img[..., :-1, :]
        pixel_dif2 = img[..., :, 1:] - img[..., :, :-1]
        reduce_axes = (-3, -2, -1)
        res1 = pixel_dif1.abs().sum(dim=reduce_axes)
        res2 = pixel_dif2.abs().sum(dim=reduce_axes)
        res = res1 + res2
    elif len(img.shape) == 5:
        pixel_dif0 = img[..., 1:, :, :] - img[...,:-1, :, :]
        pixel_dif1 = img[..., :, 1:, :] - img[...,:, :-1, :]
        pixel_dif2 = img[..., :, :, 1:] - img[...,:, :, :-1]
        reduce_axes = (-4, -3, -2, -1)
        res0 = pixel_dif0.abs().sum(dim=reduce_axes)
        res1 = pixel_dif1.abs().sum(dim=reduce_axes)
        res2 = pixel_dif2.abs().sum(dim=reduce_axes)
        res = res0 + res1 + res2

    return res



class TotalVariation(nn.Module):
    def forward(self, img) -> torch.Tensor:
        return total_variation(img)

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

        # self.mapper = LookUpTableMapper(
        #         input_dim = 8000000,  #5000**2,
        #         output_dim = 1,
        #     )

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
                norm=Norm.BATCH,
                dropout=0.5,
            ), 
            # nn.Sigmoid()  
        )

        # self.discrim = nn.Sequential(
        #     # DenseNet121(
        #     #     spatial_dims=2, 
        #     #     in_channels=1, 
        #     #     out_channels=1,
        #     #     pretrained=False, 
        #     #     dropout_prob=0.5
        #     # ),
        #     # nn.Sigmoid(),
        #     SEResNet50(
        #         spatial_dims=2, 
        #         in_channels=1,
        #         pretrained=False, 
        #     )
        # )

        # self.discrim = Discriminator(
        #     hparams.shape, 
        #     channel_multiplier=2
        # )

        replace_conv2d(self.discrim, "self.discrim")
        
        

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
        viewed_data = (self.gen[1].forward(image3d, mapped_data, cameras=cameras))
        return viewed_data, mapped_data

    def training_step(self, batch, batch_idx, optimizer_idx, stage: Optional[str]='train'):
        image3d, image2d = batch["image3d"], batch["image2d"]
        # generate images
        with torch.no_grad():
            self.varcams = RandomCameras(batch_size=self.batch_size, random=True)
        viewed_, mapped_ = self.forward(image3d, self.varcams, factor=LAMBDA)

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
                for name, hist in self.gen[0].named_parameters():
                    if name=="scaler":
                        tensorboard.add_histogram(f'{stage}_{name}', hist, self.current_epoch)

        # train generator
        if optimizer_idx == 0:
            g_loss = self.gen_step(fake_images=viewed_, real_images=image2d)
            self.log(f'{stage}_g_loss', g_loss, on_step=True, prog_bar=True, logger=True)
            self.log(f'{stage}_scaler', (self.gen[0].scaler.detach()), on_step=True, prog_bar=True, logger=True)
            # return {'loss': g_loss}  
            r_loss = 1e+1 * nn.L1Loss()(mapped_[:,[0]], image3d)   \
                   + 1e-5 * TotalVariation()(mapped_[:,[1]])
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
        viewed_, mapped_ = self.forward(image3d, self.detcams, factor=LAMBDA)
        
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
                torchvision.utils.save_image(viewed_, 
                    os.path.join(self.logdir, f'{stage}_project_{self.current_epoch}.png'))
                for name, hist in self.gen[0].named_parameters():
                    if name=="scaler":
                        tensorboard.add_histogram(f'{stage}_{name}', hist, self.current_epoch)

        g_loss = self.gen_step(fake_images=viewed_, real_images=image2d)
        d_loss = self.discrim_step(fake_images=viewed_, real_images=image2d)
        # return {"g_loss": g_loss, "d_loss": d_loss}
        r_loss = 1e+1 * nn.L1Loss()(mapped_[:,[0]], image3d)   \
               + 1e-5 * TotalVariation()(mapped_[:,[1]])
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
        
        sch_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=10)
        sch_d = torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=10)
        return [opt_g, opt_d], [sch_g, sch_d]
        # return (
        #     {'optimizer': opt_g, 'frequency': 2},
        #     {'optimizer': opt_d, 'frequency': 1}
        # )


from op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d
import math

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        # print(input.shape)
        # if len(input.shape) == 1:
            # input = input.unsqueeze(0)
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)

def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k

class Upsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)

        return out

class Downsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=1, down=self.factor, pad=self.pad)

        return out

class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer('kernel', kernel)

        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)

        return out

class EqualConv2d(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1])
        )
        self.scale = 1 / math.sqrt(in_channels * kernel_size[0] * kernel_size[1])

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))

        else:
            self.bias = None

    def forward(self, input):
        out = F.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},'
            f' {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})'
        )

class EqualConv3d(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1], kernel_size[2])
        )
        self.scale = 1 / math.sqrt(in_channels * kernel_size[0] * kernel_size[1] * kernel_size[2])

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))

        else:
            self.bias = None

    def forward(self, input):
        out = F.conv3d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},'
            f' {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})'
        )

class EqualLinear(nn.Module):
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)

        else:
            out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})'
        )

class ScaledLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()

        self.negative_slope = negative_slope

    def forward(self, input):
        out = F.leaky_relu(input, negative_slope=self.negative_slope)

        return out * math.sqrt(2)

class ModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        demodulate=True,
        upsample=False,
        downsample=False,
        blur_kernel=[1, 1, 1, 1],
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )

        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

        self.demodulate = demodulate

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, '
            f'upsample={self.upsample}, downsample={self.downsample})'
        )

    def forward(self, input, style):
        batch, in_channel, height, width = input.shape

        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
        weight = self.scale * self.weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )

        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)

        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        else:
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=self.padding, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out

class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()

        return image + self.weight * noise

class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out

class StyledConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        upsample=False,
        blur_kernel=[1, 1, 1, 1],
        demodulate=True,
    ):
        super().__init__()

        self.conv = ModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
        )

        self.noise = NoiseInjection()
        # self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))
        # self.activate = ScaledLeakyReLU(0.2)
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style, noise=None):
        out = self.conv(input, style)
        out = self.noise(out, noise=noise)
        # out = out + self.bias
        out = self.activate(out)

        return out

class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 1, 1, 1]):
        super().__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel)

        self.conv = ModulatedConv2d(in_channel, 1, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 1, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        # print('ToRGB', input.shape, style.shape, out.shape)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)

            out = out + skip

        return out


class Generator(nn.Module):
    def __init__(
        self,
        size,
        style_dim,
        n_mlp,
        channel_multiplier=2,
        blur_kernel=[1, 1, 1, 1],
        lr_mlp=0.01,
    ):
        super().__init__()

        self.size = size

        self.style_dim = style_dim

        layers = [PixelNorm()]
        # layers = []
        for i in range(n_mlp):
            layers.append(
                EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'
                )
            )

        self.style = nn.Sequential(*layers)

        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        self.input = ConstantInput(self.channels[4])
        self.conv1 = StyledConv(
            self.channels[4], self.channels[4], 3, style_dim, blur_kernel=blur_kernel
        )
        self.to_rgb1 = ToRGB(self.channels[4], style_dim, upsample=False)

        self.log_size = int(math.log(size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        in_channel = self.channels[4]

        for layer_idx in range(self.num_layers):
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2 ** res, 2 ** res]
            self.noises.register_buffer(f'noise_{layer_idx}', torch.randn(*shape))

        for i in range(3, self.log_size + 1):
            out_channel = self.channels[2 ** i]

            self.convs.append(
                StyledConv(
                    in_channel,
                    out_channel,
                    3,
                    style_dim,
                    upsample=True,
                    blur_kernel=blur_kernel,
                )
            )

            self.convs.append(
                StyledConv(
                    out_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel
                )
            )

            self.to_rgbs.append(ToRGB(out_channel, style_dim))

            in_channel = out_channel

        self.n_latent = self.log_size * 2 - 2

    def make_noise(self):
        device = self.input.input.device

        noises = [torch.randn(1, 1, 2 ** 2, 2 ** 2, device=device)]

        for i in range(3, self.log_size + 1):
            for _ in range(2):
                noises.append(torch.randn(1, 1, 2 ** i, 2 ** i, device=device))

        return noises

    def mean_latent(self, n_latent):
        latent_in = torch.randn(
            n_latent, self.style_dim, device=self.input.input.device
        )
        latent = self.style(latent_in).mean(0, keepdim=True)

        return latent

    def get_latent(self, input):
        return self.style(input)

    def forward(
        self,
        styles,
        return_latents=False,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
        input_is_latent=False,
        noise=None,
        randomize_noise=True,
    ):
        if not input_is_latent:
            styles = [self.style(s) for s in styles]

        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [
                    getattr(self.noises, f'noise_{i}') for i in range(self.num_layers)
                ]

        if truncation < 1:
            style_t = []

            for style in styles:
                style_t.append(
                    truncation_latent + truncation * (style - truncation_latent)
                )

            styles = style_t

        if len(styles) < 2:
            inject_index = self.n_latent

            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)

            else:
                latent = styles[0]

        else:
            if inject_index is None:
                inject_index = random.randint(1, self.n_latent - 1)

            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.n_latent - inject_index, 1)

            latent = torch.cat([latent, latent2], 1)

        out = self.input(latent)
        out = self.conv1(out, latent[:, 0], noise=noise[0])

        skip = self.to_rgb1(out, latent[:, 1])

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(
            self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
        ):
            out = conv1(out, latent[:, i], noise=noise1)
            out = conv2(out, latent[:, i + 1], noise=noise2)
            skip = to_rgb(out, latent[:, i + 2], skip)

            i += 2

        image = skip

        if return_latents:
            return image, latent

        else:
            return image, None

class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        downsample=False,
        blur_kernel=[1, 1, 1, 1],
        bias=True,
        activate=True,
    ):
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

            stride = 2
            self.padding = 0

        else:
            stride = 1
            self.padding = kernel_size // 2

        layers.append(
            EqualConv2d(
                in_channel,
                out_channel,
                (kernel_size, kernel_size),
                padding=self.padding,
                stride=stride,
                bias=bias and not activate,
            )
        )

        if activate:
            if bias:
                layers.append(FusedLeakyReLU(out_channel))

            else:
                layers.append(ScaledLeakyReLU(0.2))

        super().__init__(*layers)

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 1, 1, 1]):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, 3)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True)

        self.skip = ConvLayer(
            in_channel, out_channel, 1, downsample=True, activate=False, bias=False
        )

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)

        return out

class Discriminator(nn.Module):
    def __init__(self, size, channel_multiplier=2, blur_kernel=[1, 1, 1, 1]):
        super().__init__()

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        convs = [ConvLayer(1, channels[size], 1)]

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            convs.append(ResBlock(in_channel, out_channel, blur_kernel))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation='fused_lrelu'),
            EqualLinear(channels[4], 1),
        )

    def forward(self, input):
        out = self.convs(input)

        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)

        out = out.view(batch, -1)
        out = self.final_linear(out)

        return out
