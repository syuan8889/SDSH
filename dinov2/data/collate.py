# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import random


def collate_data_and_cast(samples_list, mask_ratio_tuple, mask_probability, dtype, n_tokens=None, mask_generator=None):
    # dtype = torch.half  # TODO: Remove
    # ys sample_list is a list of [(tuple),(...),(),...]:tuple(img_dict_after_transform,target)
    # ys samples_list[0][0]["global_crops"] is a list of imgs after global crops
    n_global_crops = len(samples_list[0][0]["global_crops"])
    n_local_crops = len(samples_list[0][0]["local_crops"])
    # ys 这里samples_list是一个包含32个元组的列表，每张图片global crops num = 2，所以 collated_global_crops是32*2=64张图片
    collated_global_crops = torch.stack([s[0]["global_crops"][i] for i in range(n_global_crops) for s in samples_list])

    collated_local_crops = torch.stack([s[0]["local_crops"][i] for i in range(n_local_crops) for s in samples_list])

    # ys TODO add text
    collated_text_per_img = []
    # for s in samples_list:
        # collated_text_per_img.append(s[1])

    B = len(collated_global_crops)
    N = n_tokens
    n_samples_masked = int(B * mask_probability)
    probs = torch.linspace(*mask_ratio_tuple, n_samples_masked + 1)
    upperbound = 0
    masks_list = []
    for i in range(0, n_samples_masked):# ys bool矩阵中，false表示没有被mask
        prob_min = probs[i]
        prob_max = probs[i + 1]
        masks_list.append(torch.BoolTensor(mask_generator(int(N * random.uniform(prob_min, prob_max)))))
        upperbound += int(N * prob_max)
    for i in range(n_samples_masked, B):
        masks_list.append(torch.BoolTensor(mask_generator(0)))

    random.shuffle(masks_list)# 我们需要混合一下，不然前面32个都是mask，后面都是没有mask的

    collated_masks = torch.stack(masks_list).flatten(1)# ys 把14*14个patch展开成196个patch
    mask_indices_list = collated_masks.flatten().nonzero().flatten()# ys 这是一个全局索引

    masks_weight = (1 / collated_masks.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(collated_masks)[collated_masks]

    return {
        "collated_global_crops": collated_global_crops.to(dtype),
        "collated_local_crops": collated_local_crops.to(dtype),
        "collated_masks": collated_masks,
        "mask_indices_list": mask_indices_list,
        "masks_weight": masks_weight,
        "upperbound": upperbound,
        "n_masked_patches": torch.full((1,), fill_value=mask_indices_list.shape[0], dtype=torch.long),
        "collated_text_per_img": collated_text_per_img
    }
