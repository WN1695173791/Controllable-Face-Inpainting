import torch
import torch.nn as nn
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings,  
    MeshRasterizer,  
    SoftPhongShader,
    TexturesUV,
    TexturesVertex
)
from pytorch3d.renderer import MeshRenderer as MeshRenderer3d


class MeshRenderer(nn.Module):
    def __init__(self,
                rasterize_fov,
                znear=0.1,
                zfar=10, 
                rasterize_size=224):
        super(MeshRenderer, self).__init__()

        R, T = look_at_view_transform(10, 0, 0) 
        
        self.cameras = FoVPerspectiveCameras(
            device='cuda', 
            # znear=znear, 
            # zfar=zfar, 
            znear=0.01, 
            zfar=50,             
            fov=rasterize_fov, 
            degrees=True, 
            R=R, 
            T=T
        )
        self.raster_settings = RasterizationSettings(
            image_size=rasterize_size, 
            blur_radius=0.0, 
            faces_per_pixel=1, 
        )
        self.lights = PointLights(
            device='cuda', 
            ambient_color=[[1.0,1,1]],
            specular_color=[[0,0,0]],
            diffuse_color=[[0,0,0]],
            location=[[0.0, 0.0, 1e5]]
        )

        self.renderer = MeshRenderer3d(
            rasterizer=MeshRasterizer(
                cameras=self.cameras, 
                raster_settings=self.raster_settings
            ),
            shader=SoftPhongShader(
                device='cuda', 
                cameras=self.cameras,
                lights=self.lights
            )
        )  

    def forward(self, vertex, tri, feat=None):
        """
        Return:
            mask               -- torch.tensor, size (B, 1, H, W)
            depth              -- torch.tensor, size (B, 1, H, W)
            features(optional) -- torch.tensor, size (B, C, H, W) if feat is not None

        Parameters:
            vertex          -- torch.tensor, size (B, N, 3)
            tri             -- torch.tensor, size (B, M, 3)
            feat(optional)  -- torch.tensor, size (B, C), features
        """
        batchsize = vertex.shape[0]
        tri = tri[None,].repeat(batchsize, 1, 1).type(torch.int32).contiguous()
        vertex[..., -1] = 10 - vertex[..., -1] # from camera space to world space

        # reconstruction images
        tex = TexturesVertex(verts_features=feat)
        # print(vertex, feat)
        meshes = Meshes(
            verts = vertex, 
            faces = tri,
            textures = tex,
            )
        render_img = self.renderer(meshes).permute(0,3,1,2)
        rgb_img = render_img[:,:3,:,:]
        mask = render_img[:,3:,:,:]

        # rgb_img[rgb_img>255] = 255.0
        # rgb_img[rgb_img<0] = 0.0
        # rgb_img = rgb_img/255*2-1
        mask[mask>0.1]=1.0
        return mask, None, rgb_img



