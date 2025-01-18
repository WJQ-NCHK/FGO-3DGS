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

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

# class Scene:
#     """
#         Scene 类用于管理场景的3D模型，包括相机参数、点云数据和高斯模型的初始化和加载
#     """
#
#     gaussians : GaussianModel
#
#     def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
#         """b
#         :param path: Path to colmap scene main folder.
#         :param args: 包含模型路径和源路径等模型参数
#         :param gaussians: 高斯模型对象，用于场景点的3D表示
#         :param load_iteration: 指定加载模型的迭代次数，如果为-1，则自动寻找最大迭代次数
#         :param shuffle: 是否在训练前打乱相机列表
#         :param resolution_scales: 分辨率比例列表，用于处理不同分辨率的相机
#         """
#         self.model_path = args.model_path
#         self.loaded_iter = None
#         self.gaussians = gaussians
#         # 检查并加载已有的训练模型
#         # 如果load_iteration不为None，则在输出文件夹下的point_cloud搜索迭代次数最大的文件夹
#         # 然后构造函数会判断args.source_path对应的文件夹是comlap输出还是blender输出，并从中读取场景信息到变量scene_info
#         # 如果loaded_iter有值，则直接读取对应的（已迭代）场景，如果没有，说明loaded_iter=None。
#         # 模型还没有进行训练，故调用GaussianModel.create_from_pcd从scene_info.point_cloud中创建模型
#         if load_iteration:
#             if load_iteration == -1:
#                 self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
#             else:
#                 self.loaded_iter = load_iteration
#             print("Loading trained model at iteration {}".format(self.loaded_iter))
#
#         self.train_cameras = {}# 用于训练的相机参数
#         self.test_cameras = {}# 用于测试的相机参数
#         # 根据数据集类型（COLMAP或Blender）加载场景信息
#         if os.path.exists(os.path.join(args.source_path, "sparse")):
#             #用于加载colmap输出的函数，也就是scene/dataset_readers中的readColmapSceneInfo
#             scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
#         elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
#             print("Found transforms_train.json file, assuming Blender data set!")
#             scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
#         else:
#             assert False, "Could not recognize scene type!"
#         # 如果是初次训练，初始化3D高斯模型；否则，加载已有模型
#         if not self.loaded_iter:
#             with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
#                 dest_file.write(src_file.read())
#             json_cams = []
#             camlist = []
#             if scene_info.test_cameras:
#                 camlist.extend(scene_info.test_cameras)
#             if scene_info.train_cameras:
#                 camlist.extend(scene_info.train_cameras)
#             for id, cam in enumerate(camlist):
#                 json_cams.append(camera_to_JSON(id, cam))
#             with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
#                 json.dump(json_cams, file)
#
#         if shuffle:
#             random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
#             random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling
#
#         self.cameras_extent = scene_info.nerf_normalization["radius"]
#         # 根据resolution_scales加载不同分辨率的训练和测试位姿
#         for resolution_scale in resolution_scales:
#             print("Loading Training Cameras")
#             self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
#             print("Loading Test Cameras")
#             self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
#
#         if self.loaded_iter:
#             self.gaussians.load_ply(os.path.join(self.model_path,
#                                                            "point_cloud",
#                                                            "iteration_" + str(self.loaded_iter),
#                                                            "point_cloud.ply"))
#         else:
#             self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
#
#     def save(self, iteration):
#         """
#                保存当前迭代下的3D高斯模型点云。
#                :param iteration: 当前的迭代次数。
#         """
#         point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
#         self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
#
#     def getTrainCameras(self, scale=1.0):
#         """
#                 获取指定分辨率比例的训练相机列表
#                 :param scale: 分辨率比例
#                 :return: 指定分辨率比例的训练相机列表
#         """
#         return self.train_cameras[scale]
#
#     def getTestCameras(self, scale=1.0):
#         return self.test_cameras[scale]
class Scene:
    """
    Scene 类用于管理场景的3D模型，包括相机参数、点云数据和高斯模型的初始化和加载
    """

    gaussians: GaussianModel

    def __init__(self, args: ModelParams, gaussians: GaussianModel, load_iteration=None, shuffle=True,
                 resolution_scales=[1.0]):
        """
        :param args: 包含模型路径和源路径等模型参数
        :param gaussians: 高斯模型对象，用于场景点的3D表示
        :param load_iteration: 指定加载模型的迭代次数，如果为-1，则自动寻找最大迭代次数
        :param shuffle: 是否在训练前打乱相机列表
        :param resolution_scales: 分辨率比例列表，用于处理不同分辨率的相机
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        # 如果 load_iteration 不为 None，则尝试加载已有模型
        if load_iteration is not None:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}  # 用于训练的相机参数
        self.test_cameras = {}  # 用于测试的相机参数

        # 根据数据集类型（COLMAP或Blender）加载场景信息
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            # 使用 COLMAP 格式加载场景信息
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            raise ValueError("Could not recognize scene type!")

        # 如果是初次训练，初始化3D高斯模型；否则，加载已有模型
        if not self.loaded_iter:
            print(f"Using input file path from source: {scene_info.ply_path}")

            # 确保输出目录存在
            os.makedirs(self.model_path, exist_ok=True)

            # 初始化相机 JSON 配置文件并保存
            json_cams = []
            camlist = scene_info.train_cameras + scene_info.test_cameras
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        # 如果 shuffle 为 True，打乱训练和测试相机列表
        if shuffle:
            random.shuffle(scene_info.train_cameras)
            random.shuffle(scene_info.test_cameras)

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        # 根据 resolution_scales 加载不同分辨率的训练和测试位姿
        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale,
                                                                            args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale,
                                                                           args)

        # 如果已有模型，则加载已迭代的点云文件；否则，初始化高斯模型
        if self.loaded_iter:
            point_cloud_file = os.path.join(self.model_path, "point_cloud", f"iteration_{self.loaded_iter}",
                                            "point_cloud.ply")
            self.gaussians.load_ply(point_cloud_file)
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        """
        保存当前迭代下的3D高斯模型点云。
        :param iteration: 当前的迭代次数。
        """
        point_cloud_path = os.path.join(self.model_path, f"point_cloud/iteration_{iteration}")
        os.makedirs(point_cloud_path, exist_ok=True)
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        """
        获取指定分辨率比例的训练相机列表
        :param scale: 分辨率比例
        :return: 指定分辨率比例的训练相机列表
        """
        return self.train_cameras.get(scale, [])

    def getTestCameras(self, scale=1.0):
        """
        获取指定分辨率比例的测试相机列表
        :param scale: 分辨率比例
        :return: 指定分辨率比例的测试相机列表
        """
        return self.test_cameras.get(scale, [])