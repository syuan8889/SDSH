import numpy as np
import math
import torch.utils.data as util_data
from torchvision import transforms
import torch
import six
import lmdb
import pickle
import os.path as osp
from PIL import Image
from tqdm import tqdm
from timm.data import create_transform
import torchvision.datasets as dsets


draw_range = [1, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500,
              9000, 9500, 10000]

def pr_curve(rF, qF, rL, qL, draw_range, Gnd=None, Rank=None):
    #  https://blog.csdn.net/HackerTom/article/details/89425729
    n_query = qF.shape[0]
    if Gnd is None:
        Gnd = (np.dot(qL, rL.transpose()) > 0).astype(np.float32)
    if Rank is None:
        Rank = np.argsort(CalcHammingDist(qF, rF))
    P, R = [], []
    for k in tqdm(draw_range):
        p = np.zeros(n_query)
        r = np.zeros(n_query)
        for it in range(n_query):
            gnd = Gnd[it]
            gnd_all = np.sum(gnd)
            if gnd_all == 0:
                continue
            asc_id = Rank[it][:k]
            gnd = gnd[asc_id]
            gnd_r = np.sum(gnd)
            p[it] = gnd_r / k
            r[it] = gnd_r / gnd_all
        P.append(np.mean(p))
        R.append(np.mean(r))
    return P, R


def get_precision_on_recall(rF, qF, rL, qL, recall_range, Gnd=None, Rank=None):
    #  https://blog.csdn.net/HackerTom/article/details/89425729
    n_query = qF.shape[0]
    if Gnd is None:
        Gnd = (np.dot(qL, rL.transpose()) > 0).astype(np.float32)
    if Rank is None:
        Rank = np.argsort(CalcHammingDist(qF, rF))

    p = np.zeros((n_query, len(recall_range)))
    for it in tqdm(range(n_query)):
        gnd = Gnd[it]
        gnd_r = gnd[Rank[it]]
        gnd_index = np.asarray(np.where(gnd_r == 1))[0]
        gnd_all = np.sum(gnd)
        if gnd_all == 0:
            continue
        for r_id, (r) in enumerate(recall_range):
            thres = r * gnd_all
            k = gnd_index[math.ceil(thres)-1]
            p[it, r_id] = thres / (k+1)
    return p.mean(0).tolist()


def one_hot(arr, idx):
    arr[idx] = 1
    return arr.astype(np.int32)


# TODO, qss
def compute_result(dataloader, net, device, ret_idx=False):
    bs, clses, idx = [], [], []
    net.eval()
    with torch.no_grad():
        for img, cls, ind in tqdm(dataloader):
            clses.append(cls)
            idx.append(ind)
            bs.append((net(img.to(device))))

    if ret_idx:
        return torch.cat(bs).sign(), torch.cat(clses), torch.cat(idx)
    else:
        return torch.cat(bs).sign(), torch.cat(clses)


def CalcHammingDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH


def CalcHammingDist_CUDA(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - torch.matmul(B1, B2.t()))
    return distH


def CalcTopMap(rB, qB, retrievalL, queryL, topk, ret_mtx=False):
    num_query = queryL.shape[0]
    dist_mtx = CalcHammingDist(qB, rB)
    gnd_mtx = (np.dot(queryL, retrievalL.transpose()) > 0).astype(np.float32)
    ind_mtx = np.argsort(dist_mtx, axis=1)

    topkmap = 0
    for iter in tqdm(range(num_query)):
        # gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        # hamm = CalcHammingDist(qB[iter, :], rB)
        # ind = np.argsort(hamm)
        # ind = np.argsort(dist_mtx[iter, :])
        ind = ind_mtx[iter, :]
        gnd = gnd_mtx[iter, ind]

        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd).astype(int)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    if not ret_mtx:
        return topkmap
    else:
        return topkmap, dist_mtx, gnd_mtx, ind_mtx


def CalcTopMap_CUDA(rB, qB, retrievalL, queryL, topk, ret_mtx=False):
    num_query = queryL.shape[0]
    dist_mtx = CalcHammingDist_CUDA(qB, rB)
    tmp = torch.matmul(queryL, retrievalL.t())
    gnd_mtx = tmp.gt(0).float()
    # gnd_mtx = (np.dot(queryL, retrievalL.transpose()) > 0).astype(np.float32)
    _, ind_mtx = torch.sort(dist_mtx, dim=1)
    # ind_mtx = np.argsort(dist_mtx, axis=1)

    topkmap = 0
    for iter in tqdm(range(num_query)):
        # gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        # hamm = CalcHammingDist(qB[iter, :], rB)
        # ind = np.argsort(hamm)
        # ind = np.argsort(dist_mtx[iter, :])
        ind = ind_mtx[iter, :]
        gnd = gnd_mtx[iter, ind]

        tgnd = gnd[0:topk]
        tsum = torch.sum(tgnd).int()
        if tsum.item() == 0:
            continue
        count = torch.linspace(1, tsum.item(), tsum.item(), device=rB.device)

        # tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        tindex = torch.nonzero(tgnd.eq(1)).add(1.0).squeeze()
        # topkmap_ = np.mean(count / (tindex))
        topkmap_ = torch.mean(torch.div(count, tindex))
        topkmap = topkmap + topkmap_.item()
    topkmap = topkmap / num_query
    if not ret_mtx:
        return topkmap
    else:
        return topkmap, dist_mtx, gnd_mtx, ind_mtx


def CalcTopMap_XOR(rB, qB, retrievalL, queryL, topk, ret_mtx=False, same_train_test=False, preprocessed=False,
                   binary=True):
    """
       :param qB: {0,+1}^{mxq} query bits
       :param rB: {0,+1}^{nxq} retrieval bits
       :param queryL: {0,C} query label
       :param retrievalL: {0,C} retrieval label
       :return: mAP, std of AP
    """
    if not preprocessed and binary:
        # rB[rB <= 0.] = 0
        # rB[rB > 0.] = 1
        # qB[qB <= 0.] = 0
        # qB[qB > 0.] = 1
        rB = compress_binary_to_uint(rB)
        qB = compress_binary_to_uint(qB)

    num_query = queryL.shape[0]
    mAP = np.zeros((num_query,))

    gnd_mtx = (np.dot(queryL, retrievalL.transpose()) > 0).astype(np.float32)
    dist_mtx = np.zeros_like(gnd_mtx)
    ind_mtx = np.zeros_like(dist_mtx)

    for it in tqdm(range(num_query)):
        # gnd : check if exists any db items with same label
        gnd = gnd_mtx[it, :]
        if same_train_test:
            gnd[it] = 0
        # tsum number of items with same label
        tsum = np.sum(gnd).astype(int)
        if tsum == 0:
            continue

        # sort gnd by hamming dist
        if binary:
            dist = calculate_distance(qB[it, :], rB)
        else:
            dist = calculate_distance(qB[it, :], rB, cat='euclidean')
        dist_mtx[it] = dist

        ind = np.argsort(dist)
        ind_mtx[it] = ind
        gnd = gnd_mtx[it, ind]

        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd).astype(int)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))

        mAP[it] = topkmap_

    if ret_mtx:
        return mAP.mean(), dist_mtx, gnd_mtx, ind_mtx
    else:
        return mAP.mean()

def CalcTopMap_XOR_offline(rB, qB, retrievalL, queryL, topk, same_train_test=False):
    """
       :param qB: {0,+1}^{mxq} query bits
       :param rB: {0,+1}^{nxq} retrieval bits
       :param queryL: {0,C} query label
       :param retrievalL: {0,C} retrieval label
       :return: mAP, mAP per class
    """

    num_query = queryL.shape[0]
    mAP = np.zeros((num_query,))

    class_num = int(np.max(np.unique(queryL))+1)
    mAP_per_class = np.zeros((class_num,))
    count_per_class = np.zeros_like(mAP_per_class)

    # gnd_mtx = (np.dot(queryL, retrievalL.transpose()) > 0).astype(np.float32)
    # dist_mtx = np.zeros_like(gnd_mtx)
    # ind_mtx = np.zeros_like(dist_mtx)

    for it in tqdm(range(num_query)):
        # gnd : check if exists any db items with same label
        gnd = (queryL[it] == retrievalL).astype(np.float32)
        count_per_class[int(queryL[it])] += 1
        if same_train_test:
            gnd[it] = 0
        # tsum number of items with same label
        tsum = np.sum(gnd).astype(int)
        if tsum == 0:
            continue

        # sort gnd by hamming dist
        dist = calculate_distance(qB[it, :], rB)

        ind = np.argsort(dist)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd).astype(int)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))

        mAP[it] = topkmap_
        mAP_per_class[int(queryL[it])] += topkmap_

    mAP_per_class = mAP_per_class / (count_per_class + 1.)

    return mAP.mean(), mAP_per_class


def get_precision_recall_XOR(qB, rB, queryL, retrievalL, recall_range, same_train_test=False):
    """
       :param qB: {0,+1}^{mxq} query bits
       :param rB: {0,+1}^{nxq} retrieval bits
       :param queryL: {0,C} query label
       :param retrievalL: {0,C} retrieval label
       :return: precision, recall
    """
    num_query = queryL.shape[0]

    recall_1 = np.linspace(0.05, 0.09, 3)
    recall_2 = np.linspace(0.1, 1.0, 19)
    recall_pr = np.concatenate([recall_1, recall_2])
    precision_pr = np.zeros([recall_pr.size, num_query])

    recall_topk = np.zeros((len(recall_range), num_query))
    precision_topk = np.zeros_like(recall_topk)

    for it in tqdm(range(num_query)):
        # gnd : check if exists any db items with same label
        gnd = (queryL[it] == retrievalL).astype(np.float32)
        if same_train_test:
            gnd[it] = 0
        # tsum number of items with same label
        tsum = np.sum(gnd)
        if tsum == 0:
            continue

        # sort gnd by hamming dist
        dist = calculate_distance(qB[it, :], rB)

        ind = np.argsort(dist)
        gnd = gnd[ind]

        tindex = np.asarray(np.where(gnd == 1)).squeeze(0) + 1.0


        for i, r in enumerate(recall_pr):
            recall_num = int(math.ceil(tsum*r))
            cur_idx = tindex[recall_num-1]
            precision_pr[i, it] = recall_num / cur_idx

        for i, r in enumerate(recall_range):
            t_num = np.sum(gnd[:r])
            precision_topk[i, it] = t_num / r
            recall_topk[i, it] = t_num/ tsum

    return precision_pr.mean(1), recall_pr, precision_topk.mean(1), recall_topk.mean(1)

########################################################################################################################
# the following codes are implemented for metric evaluations, above is used for training
########################################################################################################################
# count the number of '1' in the each bit of an uint32
def count_ones(xor_res):
    # the input is unsigned integer type
    xor_res = np.bitwise_and(xor_res, 0x55555555) + np.bitwise_and(np.right_shift(xor_res, 1), 0x55555555)
    xor_res = np.bitwise_and(xor_res, 0x33333333) + np.bitwise_and(np.right_shift(xor_res, 2), 0x33333333)
    xor_res = np.bitwise_and(xor_res, 0x0f0f0f0f) + np.bitwise_and(np.right_shift(xor_res, 4), 0x0f0f0f0f)
    xor_res = np.bitwise_and(xor_res, 0x00ff00ff) + np.bitwise_and(np.right_shift(xor_res, 8), 0x00ff00ff)
    xor_res = np.bitwise_and(xor_res, 0x0000ffff) + np.bitwise_and(np.right_shift(xor_res, 16), 0x0000ffff)

    return xor_res


def calculate_distance(qB, rB, cat='hamming'):
    if cat == 'hamming':  # qB and rB are uint32 type, 1*(bit_num//32), N*(bit_num//32)
        res = np.bitwise_xor(qB, rB)
        res = count_ones(res)
        distH = res.sum(1)
    elif cat == 'euclidean':
        dif = qB - rB
        distH = np.sqrt(np.sum(np.square(dif), 1))
    else:
        raise(RuntimeError('Unsupported distance category!'))

    return distH


# every 32 bits are compressed into one (unsigned) integer; input & output: numpy array
def compress_binary_to_uint(hash_codes):
    print("compressing the bit string into integers...")
    if np.min(hash_codes) < 0:
        hash_codes[hash_codes <= 0] = 0
        # hash_codes[hash_codes > 0] = 1
    hash_codes = hash_codes.astype(np.uint32)
    num = hash_codes.shape[0]
    bits = hash_codes.shape[1]
    int_bits = 32
    assert bits % int_bits == 0, 'the bit num cannot be divided by that of an integer'
    comp_hash_codes = np.zeros((num, bits // int_bits), dtype=np.uint32)  # set int32 or uint32
    for m in range(bits):
        tmp = int(0)
        tmp = np.bitwise_or(tmp, hash_codes[:, m])
        tmp = np.left_shift(tmp, int_bits - (m % int_bits) - 1)
        comp_hash_codes[:, int(m/int_bits)] = np.bitwise_or(comp_hash_codes[:, int(m/int_bits)], tmp)
    return comp_hash_codes
