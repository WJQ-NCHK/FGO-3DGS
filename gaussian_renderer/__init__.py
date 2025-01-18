#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings,GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

# 通过将高斯分布的点投影到2D屏幕上来生成渲染图像
def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    viewpoint_camera: scene.cameras.Camera类的实例
	pipe: 流水线，规定了要干什么
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    #创建一个与输入点云（高斯模型）大小相同的零张量，用于记录屏幕空间中点的位置。
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        #尝试保留张量的梯度。这是为了确保可以在反向传播过程中计算对于屏幕空间坐标的梯度
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    #计算视场的tan值，用于设置光栅化配置
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    #设置光栅化的配置，包括图片的大小，视场的tan值，背景颜色，视图矩阵，投影矩阵
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        #W2C矩阵
        viewmatrix=viewpoint_camera.world_view_transform,
        #世界到图像的投影矩阵
        projmatrix=viewpoint_camera.full_proj_transform,
        #目前的球谐函数的阶数
        sh_degree=pc.active_sh_degree,
        #相机中心位置
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )
    #创建一个高斯光栅化器对象，用于将高斯分布投影到屏幕上
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    #获取高斯分布的三维坐标，屏幕空间坐标和透明度
    means3D = pc.get_xyz
    #各个Gaussian的中心投影到图像中的坐标
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    #如果提供了预先计算的3D协方差矩阵，则使用它。否则，他将有光栅化器根据尺度和旋转进行计算
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)#获取局计算的三维协方差矩阵
    else:#获取缩放和旋转信息（对应3D高斯的协方差矩阵）
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    #如果提供了预先计算的颜色，则使用它们。
    # 否则，如果希望在python中从球谐函数中计算颜色，请执行此操作，则颜色将通过光栅化器进行从球谐函数到RGB的转换
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            #将SH特征的形状调整为（batch_size*num_points,3,(max_sh_degree+1)**2）
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            #计算相机中心到每个点的方向向量，并归一化（找到高斯中心到相机的光线在单位球体上的坐标）
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            #使用sh特征将方向向量转换为RGB颜色（球谐的傅里叶系数转换为RGB颜色）
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            #将RGB颜色范围限制在0-1之间
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    #调用光栅化器，将高斯分布投影到屏幕上，获得渲染图像和每个高斯分布在屏幕上的半径
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    #返回一个字典，包含渲染的图像，屏幕空间坐标，可见性过滤器（根据半径判断是否可见）以及每个高斯分布在屏幕上的半径
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}
