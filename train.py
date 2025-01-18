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
import torch
from random import randint
import torch.nn as nn
from typing import cast

import CaputoFO
from utils.loss_utils import l1_loss, ssim, fractional_edge_loss, caputo_fractional_derivative_fast
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr,lpip
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from Caputo_ABC import CaputoABC
from CaputoFO import CaputoFO
try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

total_elapsed_time = 0.0

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):

    first_iter = 0  # 初始化迭代参数
    tb_writer = prepare_output_and_logger(dataset)   # 设置tensorboard的写入器和日志
    gaussians = GaussianModel(dataset.sh_degree).to("cuda")  # 创建一个 GaussianModel 类的实例，输入一系列参数，其参数取自数据集。
    scene = Scene(dataset,
                  gaussians)  # （这个类的主要目的是处理场景的初始化、保存和获取相机信息等任务，）创建一个 Scene 类的实例，使用数据集和之前创建的 GaussianModel 实例作为参数。
    gaussians.training_setup(opt)  # 设置 GaussianModel 的训练参数

    #使用DataParallel进行多GPU训练
    # gaussians = nn.DataParallel(gaussians,device_ids=[1])
    # gaussians = cast(GaussianModel,gaussians.module)


    if checkpoint:  # 如果提供了检查点路径
        (model_params, first_iter) = torch.load(checkpoint)  # 通过 torch.load(checkpoint) 加载检查点的模型参数和起始迭代次数。
        gaussians.restore(model_params, opt)  # 通过 gaussians.restore 恢复模型的状态
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]  # 设置背景颜色，根据数据集是否有白色背景来选择
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")  # 将背景颜色转化为 PyTorch Tensor，并移到 GPU 上

    # 创建两个 CUDA 事件，用于测量迭代时间。
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    # 创建一个 tqdm 进度条，用于显示训练进度。
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    # 开始迭代训练
    for iteration in range(first_iter, opt.iterations + 1):
        """
            这段代码是一个循环，在每次迭代中，它首先检查GUI是否连接。如果连接了GUI，则尝试接收GUI发送的消息。
            在接收消息后，它会根据消息中的指示执行相应的操作。如果需要进行训练，
            它会在收到相应指令后执行训练，并根据需要中断训练循环。如果发生了异常，
            它会将GUI连接设置为None，以便在下一次迭代中重新尝试连接。
        """
        iter_start.record()
        if network_gui.conn == None:  # 检查 GUI 是否连接，如果连接则接收 GUI 发送的消息
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2,
                                                                                                               0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()  # 测量迭代时间

        gaussians.update_learning_rate(iteration)  # 进行学习率更新

        # Every 1000 its we increase the levels of SH up to a maximum degree
        # 每1000次迭代增加球谐函数的阶数
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera（随机选择一个训练相机）
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        num_cameras = len(viewpoint_stack)
        # viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        viewpoint_cam = viewpoint_stack[iteration % num_cameras]
        # 如果达到调试起始点，启用调试模式
        if (iteration - 1) == debug_from:
            pipe.debug = True
        # 根据设置决定是否使用随机背景颜色
        bg = torch.rand(3, device="cuda") if opt.random_background else background
        torch.cuda.empty_cache()


        # 渲染当前视角的图像
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], \
        render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        #将名为 viewpoint_cam 的视角相机对象中的原始图像转移到 GPU 上
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        lambda_dssim=0.2
        alpha = 0.1
        frac_grad = caputo_fractional_derivative_fast((image-gt_image),alpha)
        frac_grad_loss = torch.mean(frac_grad)
        edge_loss= fractional_edge_loss(image, gt_image)
            # loss = (1.0 - opt.lambda_dssim) * (Ll1**alpha) + opt.lambda_dssim * ((1.0-ssim(image, gt_image))**alpha)
            # 计算损失函数（L1 loss 和 SSIM loss）
            # loss = (1 - lambda_dssim) * (Ll1**alpha) + lambda_dssim * (1.0 - (ssim(image, gt_image) ** alpha))
        # loss = (1 - lambda_dssim) * Ll1 + lambda_dssim * (1.0 - ssim(image, gt_image))
        loss = (1-lambda_dssim)*Ll1+lambda_dssim*(1.0 - ssim(image, gt_image))+frac_grad_loss+0.1*edge_loss
        loss.backward()

        iter_end.record()  # 用于测量迭代时间。

        with torch.no_grad():  # 记录损失的指数移动平均值，并定期更新进度条。
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # 定期记录训练数据并保存模型

            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end)/1000,
                            testing_iterations, scene, render, (pipe, background))
            if iteration in saving_iterations:  # 如果达到保存迭代次数，保存场景
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification在一定的迭代次数内进行密集化处理。
            """
            这段代码是在一定的迭代次数内进行密集化处理（Densification）。在迭代次数小于opt.densify_until_iter(15000)之前，
            首先更新最大半径以进行后续的修剪操作。然后，在达到opt.densify_from_iter(500)后，
            每隔opt.densification_interval(100)次迭代就会执行一次密集化和修剪操作。
            在执行密集化和修剪操作时，会根据指定的参数调用gaussians.densify_and_prune()方法。
            该方法会根据指定的阈值对高斯核进行密集化，并根据其他条件进行修剪操作。
            此外，在达到opt.opacity_reset_interval次迭代或者当数据集的背景为白色并且迭代次数等于opt.densify_from_iter时，
            会调用gaussians.reset_opacity()方法重置高斯核的不透明度。
            """
            #在一定的迭代次数内进行密集化处理
            if iteration < opt.densify_until_iter:#在达到指定的迭代次数之前执行以下操作
                # Keep track of max radii in image-space for pruning
                #将每个像素位置上的最大半径记录在 max_radii2D 中。这是为了密集化时进行修剪（pruning）操作时的参考。
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                     radii[visibility_filter])
                # print("requires grad for viewspace_point_tensor:",viewspace_point_tensor.requires_grad)
                # print("viewspace point tensor gradient before densification stata:",viewspace_point_tensor.grad)
                #将与密集化相关的统计信息添加到 gaussians 模型中，包括视图空间点和可见性过滤器。
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                # 在指定的迭代次数之后，每隔一定的迭代间隔进行以下密集化操作。
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    # 根据当前迭代次数设置密集化的阈值。如果当前迭代次数大于 opt.opacity_reset_interval(3000)，则设置 size_threshold 为 20，否则为 None。
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    # # 将高斯分布转换为numpy数组
                    # gaussian_data = gaussians.to_numpy()
                    # #应用分数阶高斯滤波器
                    # filter_gaussian_data=fractional_gaussian_filter(gaussian_data,alpha=1.4,window_size=9,stride=4)
                    # gaussians.from_numpy(filter_gaussian_data)
                    # 执行密集化和修剪操作，其中包括梯度阈值、密集化阈值、相机范围和之前计算的 size_threshold。
                    gaussians.densify_and_prune(opt.densify_grad_threshold*1.2, 0.005, scene.cameras_extent, size_threshold)
                # 在每隔一定迭代次数或在白色背景数据集上的指定迭代次数时，执行以下操作。
                if (iteration % opt.opacity_reset_interval) == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    # 重置模型中的某些参数，涉及到透明度的操作，具体实现可以在 reset_opacity 方法中找到
                    gaussians.reset_opacity()

            # Optimizer step（执行优化器的步骤，然后清零梯度。）descriptors
            # if iteration < opt.iterations:
            #     if iteration< opt.switch_iter:
            #         gaussians.optimizer.step()
            #         gaussians.optimizer.zero_grad(set_to_none=True)
            #     else:
            #         if not isinstance(gaussians.optimizer,CaputoFO):
            #             l = gaussians.optimizer.param_groups
            #             gaussians.optimizer=CaputoFO(l)
            #         gaussians.optimizer.step()
            #         gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
            # 如果达到检查点迭代次数，保存检查点。

            if iteration in checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
        # gaussians.scheduler.step()

def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc,
                    renderArgs):

    if tb_writer:#将L1 Loss、总体Loss和迭代时间写入TensorBoard
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    #在指定的测试迭代次数，进行渲染并计算Li Loss和PSNR
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train',
                               'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                                           range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                lpips_test = 0.0
                ssim_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    #获得渲染结果和真实图像
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):#在TensorBoard中记录渲染结果和真实图像
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                             image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                                 gt_image[None], global_step=iteration)
                    #计算L1 Loss和PSNR
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    lpips_test += lpip(image, gt_image).mean().double()
                    ssim_test += ssim(image,gt_image).mean().double()
                #计算平均L1 Loss和PSNR
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])
                #在控制台打印评估结果
                ssim_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} LPIPS {} SSIM {}".format(iteration, config['name'], l1_test, psnr_test,lpips_test,ssim_test))
               #在TensorBoard中记录评估结果
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', lpips_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
        #在TensorBoard中记录场景的不透明度脂肪图和总点数
        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()#清理GPU内存
# def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc,
#                     renderArgs,total_elapsed_time):
#     total_elapsed_time += elapsed
#     if tb_writer:  # 将L1 Loss、总体Loss和迭代时间写入TensorBoard
#         tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), total_elapsed_time)
#         tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), total_elapsed_time)
#         tb_writer.add_scalar('iter_time', elapsed, total_elapsed_time)
#
#     # Report test and samples of training set
#     # 在指定的测试迭代次数，进行渲染并计算Li Loss和PSNR
#     if iteration in testing_iterations:
#         torch.cuda.empty_cache()
#         validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
#                               {'name': 'train',
#                                'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
#                                            range(5, 30, 5)]})
#
#         for config in validation_configs:
#             if config['cameras'] and len(config['cameras']) > 0:
#                 l1_test = 0.0
#                 psnr_test = 0.0
#                 lpips_test = 0.0
#                 ssim_test = 0.0
#                 for idx, viewpoint in enumerate(config['cameras']):
#                     # 获得渲染结果和真实图像
#                     image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
#                     gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
#                     if tb_writer and (idx < 5):  # 在TensorBoard中记录渲染结果和真实图像
#                         tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
#                                              image[None], global_step=total_elapsed_time)
#                         if iteration == testing_iterations[0]:
#                             tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
#                                                  gt_image[None], global_step=total_elapsed_time)
#                     # 计算L1 Loss和PSNR
#                     l1_test += l1_loss(image, gt_image).mean().double()
#                     psnr_test += psnr(image, gt_image).mean().double()
#                     lpips_test += lpip(image, gt_image).mean().double()
#                     ssim_test += ssim(image, gt_image).mean().double()
#                 # 计算平均L1 Loss和PSNR
#                 psnr_test /= len(config['cameras'])
#                 l1_test /= len(config['cameras'])
#                 lpips_test /= len(config['cameras'])
#                 # 在控制台打印评估结果
#                 ssim_test /= len(config['cameras'])
#                 print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} LPIPS {} SSIM {}".format(iteration, config['name'],
#                                                                                          l1_test, psnr_test, lpips_test,
#                                                                                          ssim_test))
#                 # 在TensorBoard中记录评估结果
#                 if tb_writer:
#                     tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, total_elapsed_time)
#                     tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, total_elapsed_time)
#                     tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', lpips_test, total_elapsed_time)
#                     tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, total_elapsed_time)
#         # 在TensorBoard中记录场景的不透明度脂肪图和总点数
#         if tb_writer:
#             tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, total_elapsed_time)
#             tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], total_elapsed_time)
#         torch.cuda.empty_cache()  # 清理GPU内存


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    #/home/ubuntu/old-gaussian-splatting/arguments/__init__.py
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000,15_000,2_0000,30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000,15_000,2_0000,30_000])
    # parser.add_argument("--test_iterations", nargs="+", type=int, default=[14_000, 60_000])
    # parser.add_argument("--save_iterations", nargs="+", type=int, default=[14_000, 60_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    # 初始化系统状态
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    ##这行代码初始化一个 GUI 服务器，使用 args.ip 和 args.port 作为参数。这可能是一个用于监视和控制训练过程的图形用户界面的一部分。
    network_gui.init(args.ip, args.port)
    # 设置pytorch是否要检测梯度计算中的异常
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    # 输入的参数包括：模型的参数（数据集的位置）、优化器的参数、其他pipeline的参数，测试迭代次数、保存迭代次数 、检查点迭代次数 、开始检查点 、调试起点
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations,
             args.checkpoint_iterations, args.start_checkpoint, args.debug_from)
    # def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from)
    # All done
    print("\nTraining complete.")