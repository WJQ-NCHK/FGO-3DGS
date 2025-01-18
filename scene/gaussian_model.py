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
import torch.distributed as dist

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
# from simple_knn import coord2Morton, boxMinMax, distBoxPoint, updateKBest
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from FactionalSGD import CaputoAdam, AlphaScheduler, Caputo_Adam, CaputoFAdam
from Caputo_ABC import CaputoABC
from CaputoFO import CaputoFO
import torch.optim.lr_scheduler as lr_scheduler

class GaussianModel(nn.Module):
    def setup_functions(self):
        """
            定义和初始化一些用于处理3D高斯模型参数的函数。
        """
        # 定义构建3D高斯协方差矩阵的函数
        # 首先从缩放及其修饰项、旋转参数得到L矩阵，将其与自己的转置相乘得到一个对称矩阵，最后用strip_symmetric函数截取其下三角不恩来节省空间。
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)# 计算实际的协方差矩阵（RSSR）
            symm = strip_symmetric(actual_covariance)# 提取对称部分
            return symm

        # 初始化一些激活函数
        self.scaling_activation = torch.exp # 用exp函数确保尺度参数非负
        self.scaling_inverse_activation = torch.log # 尺度参数的逆激活函数，用于梯度回传

        self.covariance_activation = build_covariance_from_scaling_rotation # 协方差矩阵的激活函数

        self.opacity_activation = torch.sigmoid # 用sigmoid函数确保不透明度在0到1之间
        self.inverse_opacity_activation = inverse_sigmoid # 不透明度的逆激活函数

        self.rotation_activation = torch.nn.functional.normalize # 用于标准化旋转参数的函数


    def __init__(self, sh_degree : int):
        """
            初始化3D高斯模型的参数。
            :param sh_degree: 球谐函数的最大次数，用于控制颜色表示的复杂度。
        """
        super(GaussianModel, self).__init__()
        #torch.empty(0)创建了一个形状为空的张量，即零维张量
        # 初始化球谐次数和最大球谐次数
        self.active_sh_degree = 0 #球谐函数的阶数
        self.max_sh_degree = sh_degree # 允许的最大球谐次数
        # 初始化3D高斯模型的各项参数
        self._xyz = nn.Parameter(torch.empty(0).cuda(),requires_grad=True)# 3D高斯的中心位置（均值）
        self._features_dc = nn.Parameter(torch.empty(0).cuda(),requires_grad=True)# 第一个球谐系数，用于表示基础颜色
        self._features_rest = nn.Parameter(torch.empty(0).cuda(),requires_grad=True)# 其余的球谐系数，用于表示颜色的细节和变化
        self._scaling = nn.Parameter(torch.empty(0).cuda(),requires_grad=True)# 3D高斯的尺度参数，控制高斯的宽度
        self._rotation = nn.Parameter(torch.empty(0).cuda(),requires_grad=True)# 3D高斯的旋转参数，用四元数表示
        self._opacity = nn.Parameter(torch.empty(0).cuda(),requires_grad=True)# 3D高斯的不透明度（在sigmoid激活前），控制可见性
        self.max_radii2D = torch.empty(0).cuda()# 在2D投影中，每个高斯的最大半径
        self.xyz_gradient_accum = torch.empty(0).cuda()# 用于累积3D高斯中心位置的梯度，当它太大时要对Gaussian进行分裂或复制
        self.denom = torch.empty(0).cuda()# 与累积梯度配合使用，表示统计多少次累积梯度，算平均梯度时除掉这个（denom=denominator分母）
        self.optimizer = None# 优化器，用于调整上述参数以改进模型（论文采用Adam，见附录B伪代码）
        self.scheduler = None#动态更新分数阶
        self.percent_dense = 0 #初始化百分比密度为0。参与控制Gaussian密集程度的超参数
        self.spatial_lr_scale = 0 #初始化空间学习速率缩放为0，坐标的学习速率要乘上这个，抵消在不同尺度下应用同一个学习率带来的问题
        self.setup_functions()# 调用setup_functions来初始化一些处理函数

    def to_numpy(self):
        """
        Convert Gaussian attributes to numpy arrays.
        """
        return {
            "xyz": self._xyz.cpu().numpy(),
            "f_dc": self._features_dc.cpu().numpy(),
            "f_rest": self._features_rest.cpu().numpy(),
            "opacity": self._opacity.cpu().numpy(),
            "scaling": self._scaling.cpu().numpy(),
            "rotation": self._rotation.cpu().numpy()
        }

    def from_numpy(self, data):
        """
        Convert numpy arrays back to Gaussian attributes.
        """
        self._xyz = torch.from_numpy(data["xyz"]).to("cuda")
        self._features_dc = torch.from_numpy(data["f_dc"]).to("cuda")
        self._features_rest = torch.from_numpy(data["f_rest"]).to("cuda")
        self._opacity = torch.from_numpy(data["opacity"]).to("cuda")
        self._scaling = torch.from_numpy(data["scaling"]).to("cuda")
        self._rotation = torch.from_numpy(data["rotation"]).to("cuda")
    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        #增加球谐函数的阶数
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        #与学习率有关，防止固定学习率适配不同尺度的场景出现问题
        self.spatial_lr_scale = spatial_lr_scale
        #N为点的个数
        #稀疏点云的3D坐标，大小为（N，3）
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        #球谐的直流分量，大小为（N，3），RGB2SH（rgb）=(rgb - 0.5) / C0，C0=0.28209479177387814
        #pcd.colors原始范围是0-1
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        #RGB三通道球谐的所有系数，大小为（N，3,（最大球谐阶数+1）的平方）
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])
        # distCUDA2函数由simple_knn.cu的SimpleKNN::knn函数实现
        # k近邻算法，求每个点最近的k个点。
        # 算法近似knn
        # 原理是将3D空间中的每个点用莫顿编码转换为1D坐标然后对1D坐标排序从而确定每个点最近的三个点
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        #repeat(1, 3)标明三个方向上的scale的初始值相等，scales大小为（N，3）
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        #旋转矩阵大小为（N，4）
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        #初始化表示旋转矩阵的四元数为1（第0维为1,其他维度为0）
        rots[:, 0] = 1
        #不透明度在经历sigmoid前的值大小为（N，1）
        #这里将不透明初始化为0.1,但在存储的时候将0.1取其经过sigmoid前的值。故先用sigmoid的反函数求0.1的值再存储
        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        #作为参数的高斯椭球体中心坐标（N，3）
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        #RGB三个通道的直流分量（N，3,1）
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        #RGB三个通道的高阶分量（N，3,(最大球谐阶数+1)的平方-1）
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        #作为参数的缩放
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        #作为参数的旋转四元数（N，4）
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        #作为参数的不透明度（sigmoid之前）（N，1）
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        #大小为（N，）
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        # dist.barrier()  # 同步各进程
        # assert self._xyz.shape[0] > 0, "XYZ tensor not initialized correctly!"
        # print(f"Rank {dist.get_rank()} - XYZ shape: {self.get_xyz.shape}")
    #初始化训练工作
    def training_setup(self, training_args):
        #控制Gaussian的密度，在"densify_and_clone"中被使用
        self.percent_dense = training_args.percent_dense
        #坐标的累积梯度
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        #设置分别对应于各个参数的学习率
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]
        #设置根据每个参数的特性（位置、颜色、不透明度等），在初始化时传入对应的历史项数
        param_history_length = {
            "xyz": 1,  # For positions, use history length 1 (i.e., only use the current gradient)
            "scaling": 1,  # For scaling, use history length 1
            "f_dc": 1,  # For color features, use history length 10
            "f_rest": 1,  # For color features, use history length 10
            "opacity": 1,  # For opacity, use history length 10
            "rotation": 1  # For rotation, use history length 1
        }
        # l = [
        #     {'params': [self._xyz], 'lr': (training_args.position_lr_init * self.spatial_lr_scale)*2, "name": "xyz"},
        #     {'params': [self._features_dc], 'lr': (training_args.feature_lr)*2, "name": "f_dc"},
        #     {'params': [self._features_rest], 'lr': (training_args.feature_lr / 20.0)*2, "name": "f_rest"},
        #     {'params': [self._opacity], 'lr': (training_args.opacity_lr)*2, "name": "opacity"},
        #     {'params': [self._scaling], 'lr': (training_args.scaling_lr)*2, "name": "scaling"},
        #     {'params': [self._rotation], 'lr': (training_args.rotation_lr)*2, "name": "rotation"}
        # ]
        # self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        # self.optimizer = CaputoABC(l)
        self.optimizer = CaputoFO(l, lr=0.0, eps=1e-15,param_history_length=param_history_length)

        #坐标的学习率规划函数
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
    #更新Gaussian坐标的学习率
    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    #构建ply文件的健列表
    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            #self._features_dc（N，3,1）
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            #self._features_rest（N，3,（最大球谐函数阶数+1）的平方-1）
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        #self._scaling（N，3）
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        #self._rotation（N，4）
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        #所有要保存的值合并成为一个大数组
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    #重置不透明度
    def reset_opacity(self):
        #get_opacity返回经过exp的不透明度。是真的不透明度
        #让所有的不透明度不能超过0.01
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        #更新优化器中的不透明度
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    #读取ply文件并将数据转换为nn.Parameter类型等待优化
    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    # def replace_tensor_to_optimizer(self, tensor, name):
    #     #修改Adam优化器的状态变量：动量和平方动量
    #     optimizable_tensors = {}
    #     for group in self.optimizer.param_groups:
    #         if group["name"] == name:
    #             # stored_state = self.optimizer.state.get(group['params'][0], None)
    #             stored_state = self.optimizer.state.get(group['params'][0], {})
    #             #将动量清零
    #             stored_state["exp_avg"] = torch.zeros_like(tensor)
    #             #将平方动量清零
    #             stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
    #
    #             del self.optimizer.state[group['params'][0]]
    #             group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
    #             self.optimizer.state[group['params'][0]] = stored_state
    #
    #             optimizable_tensors[group["name"]] = group["params"][0]
    #     return optimizable_tensors
    def replace_tensor_to_optimizer(self, tensor, name):
        # 修改 Adam 优化器的状态变量：动量和平方动量
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                # 获取或初始化状态字典
                param = group['params'][0]
                stored_state = self.optimizer.state.get(param, {})

                # 初始化动量和平方动量为与 tensor 相同大小的零张量
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                # 删除旧的参数状态并替换为新参数
                if param in self.optimizer.state:
                    del self.optimizer.state[param]

                new_param = nn.Parameter(tensor.requires_grad_(True))
                group["params"][0] = new_param  # 替换参数
                self.optimizer.state[new_param] = stored_state  # 更新状态

                # 将新参数存储为可优化的张量
                optimizable_tensors[group["name"]] = new_param

        return optimizable_tensors

    def _prune_optimizer(self, mask):
        #根据mask裁减一部分参数及其动量和二阶动量
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    #删除Gaussian并移除对应的所有属性
    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)
        #重置各个参数
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    #将新的张量字典添加到优化器
    # def cat_tensors_to_optimizer(self, tensors_dict):
    #     optimizable_tensors = {}
    #     for group in self.optimizer.param_groups:
    #         assert len(group["params"]) == 1
    #         extension_tensor = tensors_dict[group["name"]]
    #         stored_state = self.optimizer.state.get(group['params'][0], None)
    #         if stored_state is not None:
    #
    #             stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
    #             stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)
    #
    #             del self.optimizer.state[group['params'][0]]
    #             group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
    #             self.optimizer.state[group['params'][0]] = stored_state
    #
    #             optimizable_tensors[group["name"]] = group["params"][0]
    #         else:
    #             group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
    #             optimizable_tensors[group["name"]] = group["params"][0]
    #
    #     return optimizable_tensors
    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            param = group["params"][0]

            if param in self.optimizer.state:
                stored_state = self.optimizer.state[param]

                # Ensure 'exp_avg' and 'exp_avg_sq' are initialized
                if "exp_avg" not in stored_state:
                    stored_state["exp_avg"] = torch.zeros_like(param.data)
                if "exp_avg_sq" not in stored_state:
                    stored_state["exp_avg_sq"] = torch.zeros_like(param.data)

                # Concatenate new tensors to the existing state tensors
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)),
                                                    dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                                                       dim=0)

                # Update parameter and state in optimizer
                del self.optimizer.state[param]
                new_param = nn.Parameter(torch.cat((param.data, extension_tensor), dim=0).requires_grad_(True))
                group["params"][0] = new_param
                self.optimizer.state[new_param] = stored_state

                if 'grad_buffer' in stored_state:
                    stored_state['grad_buffer'] = torch.cat(
                        (stored_state['grad_buffer'], torch.zeros_like(extension_tensor)), dim=0)
                optimizable_tensors[group["name"]] = new_param
            else:
                new_param = nn.Parameter(torch.cat((param.data, extension_tensor), dim=0).requires_grad_(True))
                group["params"][0] = new_param
                optimizable_tensors[group["name"]] = new_param

        return optimizable_tensors
    # 将新的密集化点的相关特征保存在一个字典中。
    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        #新增Gaussian，把新属性添加到优化器中
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}
        #将字典中的张量concat成可以优化的张量，具体实现可能是将字典中的每个张量进行堆叠，以便在优化器中进行处理
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        #更新模型中原始点集的相关特征，使用新的密集化后的特征
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        #重新初始化一些用于梯度计算和密集化操作的变量
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")


    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]#获取初始点的数量
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")#创建一个长度为初始点数量的梯度张量，并将计算得到的梯度填充到其中
        padded_grad[:grads.shape[0]] = grads.squeeze()
        #创建一个掩码，标记那些梯度大于等于指定阈值的点
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)

        #一步过滤掉那些缩放大于一定百分比的场景范围的点
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        #为每个点生成新的样本，其中stds是点的缩放，means是均值
        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        #使用均值和标准差生成样本
        samples = torch.normal(mean=means, std=stds)
        #为每个点构建旋转矩阵，并将其重复N次
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        #将旋转后的样本点添加到原始点的位置
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        #生成新的缩放参数
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        #将旋转矩阵重复N次
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        #将原始点的特征重复N次
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        #调用另一个方法densification_postfix,该方法对新生成的点执行后处理操作
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)
        #创建一个修建（pruning）的过滤器将新生成的点添加到原始点的掩码之后
        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        #根据修剪过滤器，修建模型中的一些参数
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        # 建一个掩码，标记满足梯度条件的点。具体来说，对于每个点，计算其梯度的L2范数，如果大于等于指定的梯度阈值，则标记为True，否则标记为False。
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        #在上述掩码的基础上，进一步过滤掉那些缩放（scaling）大于一定百分比（self.percent_dense）的场景范围（scene_extent）的点。这样可以确保新添加的点不会太远离原始数据。
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        # 根据掩码选取符合条件的点的其他特征，如颜色、透明度、缩放和旋转等。
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    # 执行密集化和修剪操作
    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        # 执行密集化和修剪操作，其中包括梯度阈值、密集化阈值、相机范围和之前计算的 size_threshold。
        """
        首先，计算每个高斯核的梯度值，将其存储在grads中。
        调用densify_and_clone方法对梯度值较大的高斯核进行复制和密集化。
        调用densify_and_split方法对梯度值较大的高斯核进行分裂和密集化。
        根据给定的条件（最小不透明度、屏幕大小限制等）生成修剪掩码prune_mask。
        调用prune_points方法根据修剪掩码对高斯核进行修剪操作。
        最后，清空CUDA缓存，释放内存。
        """
        # 计算密度估计的梯度
        grads = self.xyz_gradient_accum / self.denom
        # 将梯度中的 NaN（非数值）值设置为零，以处理可能的数值不稳定性。
        grads[grads.isnan()] = 0.0
        # grads = torch.where(self.denom !=0,self.xyz_gradient_accum/self.denom,torch.zeros_like(self.xyz_gradient_accum))
        # 对under reconstruction的区域进行稠密化和复制操作
        self.densify_and_clone(grads, max_grad, extent)
        # 对over reconstruction的区域进行稠密化和分割操作
        self.densify_and_split(grads, max_grad, extent)
        # 创建一个掩码，标记那些透明度小于指定阈值的点。.squeeze() 用于去除掩码中的单维度
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        # 如何设置了相机的范围
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size #创建一个掩码，标记在图像空间中半径大于指定阈值的点。
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent #创建一个掩码，标记在世界空间中尺寸大于指定阈值的点。
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws) #将这两个掩码与先前的透明度掩码进行逻辑或操作，得到最终的修剪掩码
        self.prune_points(prune_mask) #：根据修剪掩码，修剪模型中的一些参数

        torch.cuda.empty_cache() #清理 GPU 缓存，释放一些内存

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        #统计坐标的累积梯度和均值的分母
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1