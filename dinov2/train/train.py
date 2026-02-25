# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.
import argparse
import logging
import math
import os
from functools import partial

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from fvcore.common.checkpoint import PeriodicCheckpointer
import torch
from torchvision import transforms
from dinov2.data import SamplerType, make_data_loader, make_dataset
from dinov2.data import collate_data_and_cast, DataAugmentationDINO, MaskingGenerator
import dinov2.distributed as distributed
from dinov2.fsdp import FSDPCheckpointer
from dinov2.logging import MetricLogger
from dinov2.utils.config import setup
from dinov2.utils.utils import CosineScheduler

from dinov2.train.ssl_meta_arch import SSLMetaArch
from torch.utils.data import Dataset
from custom_dataset import CustomImageDataset
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 全局变量，用于跟踪最佳mAP和最佳迭代次数
best_map = 0.0
best_iteration = 0


torch.backends.cuda.matmul.allow_tf32 = True  # PyTorch 1.12 sets this to False by default
logger = logging.getLogger("dinov2")


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("DINOv2 training", add_help=add_help)
    parser.add_argument("--config-file", default="/dinov2/train/config.yaml", metavar="FILE", help="path to config file")
    parser.add_argument( "--no-resume",action="store_true",help="Whether to not attempt to resume from the checkpoint directory. ",)  # qss
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--eval", type=str, default="", help="Eval type to perform")
    parser.add_argument("--output-dir", default="",
                        type=str, help="Output directory to save logs and checkpoints",)  # qss
    # ys: for distribute training
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
            distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER, help="""
        Modify config options at the end of the command. For Yacs configs, use
        space-separated "PATH.KEY VALUE" pairs.
        For python-based LazyConfig, use "path.key=value".
         """.strip(), )
    return parser


def build_optimizer(cfg, params_groups):
    return torch.optim.AdamW(params_groups, betas=(cfg.optim.adamw_beta1, cfg.optim.adamw_beta2))


def build_schedulers(cfg):
    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    lr = dict(
        base_value=cfg.optim["lr"],
        final_value=cfg.optim["min_lr"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
        warmup_iters=cfg.optim["warmup_epochs"] * OFFICIAL_EPOCH_LENGTH,
        start_warmup_value=0,
    )
    wd = dict(
        base_value=cfg.optim["weight_decay"],
        final_value=cfg.optim["weight_decay_end"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
    )
    momentum = dict(
        base_value=cfg.teacher["momentum_teacher"],
        final_value=cfg.teacher["final_momentum_teacher"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
    )
    teacher_temp = dict(
        base_value=cfg.teacher["teacher_temp"],
        final_value=cfg.teacher["teacher_temp"],
        total_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
        warmup_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
        start_warmup_value=cfg.teacher["warmup_teacher_temp"],
    )

    lr_schedule = CosineScheduler(**lr)
    wd_schedule = CosineScheduler(**wd)
    momentum_schedule = CosineScheduler(**momentum)
    teacher_temp_schedule = CosineScheduler(**teacher_temp)
    last_layer_lr_schedule = CosineScheduler(**lr)

    last_layer_lr_schedule.schedule[
        : cfg.optim["freeze_last_layer_epochs"] * OFFICIAL_EPOCH_LENGTH
    ] = 0  # mimicking the original schedules

    logger.info("Schedulers ready.")

    return (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
    )


def apply_optim_scheduler(optimizer, lr, wd, last_layer_lr):
    for param_group in optimizer.param_groups:
        is_last_layer = param_group["is_last_layer"]
        # qss
        is_hash_layer = param_group["is_hash_layer"]
        lr_multiplier = param_group["lr_multiplier"]
        wd_multiplier = param_group["wd_multiplier"]
        param_group["weight_decay"] = wd * wd_multiplier
        param_group["lr"] = (last_layer_lr if is_last_layer else lr) * lr_multiplier
        # qss TODO: 需要修改
        # if is_hash_layer:
          #  param_group["lr"] *= 10.0


# qss
def do_test(cfg, model, iteration, database_data_loader=None, query_data_loader=None, database_dataset=None, query_dataset=None):
    global best_map, best_iteration
    new_state_dict = model.teacher.state_dict()
    if distributed.is_main_process():
        # 使用共享的eval目录
        eval_dir = os.path.join(cfg.train.output_dir, "eval")
        os.makedirs(eval_dir, exist_ok=True)
        
        # 图像检索测试
        logger.info("开始图像检索测试...")
        
        # 设置模型为评估模式
        model.teacher.eval()
        
        # 提取数据库图像的哈希表示和标签
        logger.info("提取数据库图像哈希表示...")
        database_hashes = []
        database_labels = []
        
        with torch.no_grad():
            for batch in database_data_loader:
                images = batch[0].cuda(non_blocking=True)
                labels = batch[1].cuda(non_blocking=True)
                
                # 获取teacher backbone输出
                hash_codes = model.teacher.backbone(images, is_training=False)
                
                # 二值化哈希码
                binary_hash = torch.sign(hash_codes)  # 将哈希码二值化为{-1, 1}
                # binary_hash = (binary_hash + 1) / 2  # 转换为{0, 1}
                
                database_hashes.append(binary_hash)
                database_labels.append(labels)
        
        # 提取查询图像的哈希表示和标签
        logger.info("提取查询图像哈希表示...")
        query_hashes = []
        query_labels = []
        
        with torch.no_grad():
            for batch in query_data_loader:
                images = batch[0].cuda(non_blocking=True)
                labels = batch[1].cuda(non_blocking=True)
                
                # 获取teacher backbone输出
                hash_codes = model.teacher.backbone(images, is_training=False)
                
                # 二值化哈希码
                binary_hash = torch.sign(hash_codes)  # 将哈希码二值化为{-1, 1}
                # binary_hash = (binary_hash + 1) / 2  # 转换为{0, 1}
                
                query_hashes.append(binary_hash)
                query_labels.append(labels)
        
        # 合并所有批次的数据
        database_hashes = torch.cat(database_hashes, dim=0)
        database_labels = torch.cat(database_labels, dim=0)
        query_hashes = torch.cat(query_hashes, dim=0)
        query_labels = torch.cat(query_labels, dim=0)
        
        # 转换为one-hot编码
        num_classes = len(database_dataset.classes)
        database_labels_onehot = torch.zeros(database_labels.size(0), num_classes, device=database_labels.device)
        query_labels_onehot = torch.zeros(query_labels.size(0), num_classes, device=query_labels.device)
        
        database_labels_onehot.scatter_(1, database_labels.unsqueeze(1), 1)
        query_labels_onehot.scatter_(1, query_labels.unsqueeze(1), 1)
        
        # 调用CalcTopMap函数计算mAP
        from dinov2.utils.retrieval_tools import CalcTopMap_CUDA
        
        logger.info(f"数据库哈希码形状: {database_hashes.shape}")
        logger.info(f"查询哈希码形状: {query_hashes.shape}")
        logger.info(f"数据库标签形状: {database_labels_onehot.shape}")
        logger.info(f"查询标签形状: {query_labels_onehot.shape}")
        
        # 计算mAP，topk设置为数据库大小表示使用所有检索结果
        topk = database_hashes.shape[0]  # 使用数据库大小作为topk
        mAP = CalcTopMap_CUDA(
            rB=database_hashes,  # 数据库哈希码
            qB=query_hashes,     # 查询哈希码
            retrievalL=database_labels_onehot,  # 数据库标签
            queryL=query_labels_onehot,         # 查询标签
            topk=topk,  # 使用所有检索结果
            ret_mtx=False
        )
        
        logger.info(f"图像检索mAP: {mAP:.4f}")
        
        # 检查是否为最佳mAP
        is_best = mAP > best_map
        if is_best:
            best_map = mAP
            best_iteration = iteration
            logger.info(f"发现新的最佳mAP: {best_map:.4f}!")
        
        # 保存当前迭代结果到文件（追加模式）
        result_file = os.path.join(eval_dir, "retrieval_results.txt")
        with open(result_file, 'a') as f:
            f.write(f"\n{'='*50}\n")
            f.write(f"图像检索测试结果 (迭代 {iteration})\n")
            f.write(f"测试时间: {os.popen('date').read().strip()}\n")
            f.write(f"mAP: {mAP:.4f}\n")
            f.write(f"是否为最佳: {'是' if is_best else '否'}\n")
            f.write(f"当前最佳mAP: {best_map:.4f}\n")
            if is_best:
                f.write(f"当前最佳模型迭代次数: {iteration}\n")
            else:
                f.write(f"当前最佳模型迭代次数: {best_iteration}\n")
            f.write(f"数据库样本数: {len(database_dataset)}\n")
            f.write(f"查询样本数: {len(query_dataset)}\n")
            f.write(f"类别数: {num_classes}\n")
            f.write(f"哈希码维度: {database_hashes.shape[1]}\n")
            f.write(f"{'='*50}\n")
        
        # 只有当是最佳模型时才保存
        if is_best:
            # 保存最佳模型到eval目录（覆盖之前的）
            teacher_ckp_path = os.path.join(eval_dir, "teacher_checkpoint.pth")
            torch.save({"teacher": new_state_dict, "mAP": best_map, "iteration": iteration}, teacher_ckp_path)
            logger.info(f"新的最佳模型已保存到: {teacher_ckp_path}")
        else:
            logger.info(f"当前mAP ({mAP:.4f}) 不是最佳，最佳mAP为{best_map:.4f}，最佳模型迭代次数为{best_iteration}，当前模型迭代次数为{iteration}，跳过模型保存")
        
        logger.info(f"检索测试完成，结果已保存到: {result_file}")
        if is_best:
            logger.info(f"最佳模型已保存到: {eval_dir}")


def do_train(cfg, model, resume=False):
    model.train()
    inputs_dtype = torch.half
    fp16_scaler = model.fp16_scaler  # for mixed precision training

    # setup optimizer

    optimizer = build_optimizer(cfg, model.get_params_groups())

    (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
    ) = build_schedulers(cfg)

    # checkpointer
    checkpointer = FSDPCheckpointer(model, cfg.train.output_dir, optimizer=optimizer, save_to_disk=True)

    start_iter = checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1

    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    max_iter = cfg.optim.epochs * OFFICIAL_EPOCH_LENGTH

    periodic_checkpointer = (PeriodicCheckpointer(
        checkpointer,
        period=3 * OFFICIAL_EPOCH_LENGTH,
        max_iter=max_iter,
        max_to_keep=1,
    ))  # qss TODO 

    # setup data preprocessing

    img_size = cfg.crops.global_crops_size
    patch_size = cfg.student.patch_size
    n_tokens = (img_size // patch_size) ** 2
    mask_generator = MaskingGenerator(
        input_size=(img_size // patch_size, img_size // patch_size),
        max_num_patches=0.5 * img_size // patch_size * img_size // patch_size,
    )

    data_transform = DataAugmentationDINO(
        cfg.crops.global_crops_scale,
        cfg.crops.local_crops_scale,
        cfg.crops.local_crops_number,
        global_crops_size=cfg.crops.global_crops_size,
        local_crops_size=cfg.crops.local_crops_size,
    )

    collate_fn = partial(
        collate_data_and_cast,
        mask_ratio_tuple=cfg.ibot.mask_ratio_min_max,
        mask_probability=cfg.ibot.mask_sample_probability,
        n_tokens=n_tokens,
        mask_generator=mask_generator,
        dtype=inputs_dtype,
    )
    # 以下需要修改，使用自己的数据集，qss
    transform_test = transforms.Compose([
        transforms.Resize(256,interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(cfg.crops.global_crops_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ])

    dataset = CustomImageDataset(
        root=cfg.train.data_path,
        train=True,
        train_ratio=cfg.train.train_ratio,
        transform=data_transform
    )

    # qss
    database_dataset = CustomImageDataset(
        root=cfg.train.data_path,
        train=True,
        train_ratio=cfg.train.train_ratio,
        transform=transform_test
    )
    query_dataset = CustomImageDataset(
        root=cfg.train.data_path,
        train=False,
        train_ratio=cfg.train.train_ratio,
        transform=transform_test
    )
    # print(dataset[0])
    # sampler_type = SamplerType.INFINITE
    sampler_type = SamplerType.SHARDED_INFINITE
    data_loader = make_data_loader(
        dataset=dataset,
        batch_size=cfg.train.batch_size_per_gpu,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        seed=cfg.train.seed,  # TODO: Fix this -- cfg.train.seed
        sampler_type=sampler_type,
        sampler_advance=start_iter * cfg.train.batch_size_per_gpu,  # TODO(qas): fix this -- start_iter * cfg.train.batch_size_per_gpu,
        drop_last=True,
        collate_fn=collate_fn,
    )

    # qss
    database_data_loader = make_data_loader(
        dataset=database_dataset,
        batch_size=cfg.train.batch_size_per_gpu,
        num_workers=cfg.train.num_workers,
        shuffle=False,
        seed=cfg.train.seed,
        sampler_type=SamplerType.EPOCH,
        sampler_advance=0,
        drop_last=False,
    )
    query_data_loader = make_data_loader(
        dataset=query_dataset,
        batch_size=cfg.train.batch_size_per_gpu,
        num_workers=cfg.train.num_workers,
        shuffle=False,
        seed=cfg.train.seed,
        sampler_type=SamplerType.EPOCH,
        sampler_advance=0,
        drop_last=False,
    )

    # training loop

    iteration = start_iter

    logger.info("Starting training from iteration {}".format(start_iter))
    metrics_file = os.path.join(cfg.train.output_dir, "training_metrics.json")
    metric_logger = MetricLogger(delimiter="  ", output_file=metrics_file)
    header = "Training"
    for data in metric_logger.log_every(
        data_loader,
        10,
        header,
        max_iter,
        start_iter,
    ):
        current_batch_size = data["collated_global_crops"].shape[0] / 2
        if iteration > max_iter:
            return

        # apply schedules
        lr = lr_schedule[iteration]
        wd = wd_schedule[iteration]
        mom = momentum_schedule[iteration]
        teacher_temp = teacher_temp_schedule[iteration]
        last_layer_lr = last_layer_lr_schedule[iteration]
        apply_optim_scheduler(optimizer, lr, wd, last_layer_lr)

        # compute losses
        optimizer.zero_grad(set_to_none=True)
        loss_dict = model.forward_backward(data, teacher_temp=teacher_temp)

        # clip gradients
        if fp16_scaler is not None:
            if cfg.optim.clip_grad:
                fp16_scaler.unscale_(optimizer)
                for v in model.student.values():
                    v.clip_grad_norm_(cfg.optim.clip_grad)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()
        else:
            if cfg.optim.clip_grad:
                for v in model.student.values():
                    v.clip_grad_norm_(cfg.optim.clip_grad)
            optimizer.step()
        # perform teacher EMA update
        model.update_teacher(mom)

        # logging
        if distributed.get_global_size() > 1:
            for v in loss_dict.values():
                torch.distributed.all_reduce(v)
        loss_dict_reduced = {k: v.item() / distributed.get_global_size() for k, v in loss_dict.items()}
        for k, v in loss_dict_reduced.items():
            if math.isinf(v) :
                logger.info("inf detected %s",k)
                raise AssertionError
            if math.isnan(v):
                logger.info("NaN detected %s",k)
                raise AssertionError

        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        metric_logger.update(lr=lr)
        metric_logger.update(wd=wd)
        metric_logger.update(mom=mom)
        metric_logger.update(last_layer_lr=last_layer_lr)
        metric_logger.update(current_batch_size=current_batch_size)
        metric_logger.update(total_loss=losses_reduced, **loss_dict_reduced)

        # checkpointing and testing, mAP test needed, qss

        if cfg.evaluation.eval_period_iterations > 0 and (iteration + 1) % cfg.evaluation.eval_period_iterations == 0:

            do_test(cfg, model, iteration, database_data_loader, query_data_loader, database_dataset, query_dataset)
            torch.cuda.synchronize()
        # periodic_checkpointer.step(iteration)

        iteration = iteration + 1
    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def main(args):
    cfg = setup(args)

    model = SSLMetaArch(cfg).to(torch.device("cuda"))
    model.prepare_for_distributed_training()



    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        iteration = (
            FSDPCheckpointer(model, save_dir=cfg.train.output_dir)
            .resume_or_load(cfg.MODEL.WEIGHTS, resume=not args.no_resume)
            .get("iteration", -1)
            + 1
        )
        # 对于eval_only模式，需要创建数据加载器
        transform_test = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(cfg.crops.global_crops_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ])
        
        database_dataset = CustomImageDataset(
            root=cfg.train.data_path,
            train=True,
            train_ratio=cfg.train.train_ratio,
            transform=transform_test
        )
        query_dataset = CustomImageDataset(
            root=cfg.train.data_path,
            train=False,
            train_ratio=cfg.train.train_ratio,
            transform=transform_test
        )
        
        database_data_loader = make_data_loader(
            dataset=database_dataset,
            batch_size=cfg.train.batch_size_per_gpu,
            num_workers=cfg.train.num_workers,
            shuffle=False,
            seed=cfg.train.seed,
            sampler_type=SamplerType.EPOCH,
            sampler_advance=0,
            drop_last=False,
        )
        query_data_loader = make_data_loader(
            dataset=query_dataset,
            batch_size=cfg.train.batch_size_per_gpu,
            num_workers=cfg.train.num_workers,
            shuffle=False,
            seed=cfg.train.seed,
            sampler_type=SamplerType.EPOCH,
            sampler_advance=0,
            drop_last=False,
        )
        
        return do_test(cfg, model, iteration, database_data_loader, query_data_loader, database_dataset, query_dataset)

    do_train(cfg, model, resume=not args.no_resume)


if __name__ == "__main__":
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    args = get_args_parser(add_help=True).parse_args()
    main(args)
