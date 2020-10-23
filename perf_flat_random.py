# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import time

def using_flat_index(d, xb, xq):
    index_flat = faiss.IndexFlatL2(d)  # build a flat (CPU) index

    # make it a flat GPU index
    gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
    #  gpu_index_flat = faiss.index_cpu_to_gpus_list(index_flat, gpus = [2], ngpu=1)

    gpu_index_flat.add(xb)         # add vectors to the index
    #  print(gpu_index_flat.ntotal)

    k = K                        # we want to see k nearest neighbors
    D, I = gpu_index_flat.search(xq, k)  # actual search
    #  print(I[:5])                   # neighbors of the 5 first queries
    #  print(I[-5:])                  # neighbors of the 5 last queries
    return D, I

def using_IVF_index(d, xb, xq):
    # the number of Voronoi cells
    # see https://github.com/facebookresearch/faiss/wiki/Faster-search
    nlist = 100

    quantizer = faiss.IndexFlatL2(d)  # the other index
    index_ivf = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
    # here we specify METRIC_L2, by default it performs inner-product search

    # make it an IVF GPU index
    gpu_index_ivf = faiss.index_cpu_to_gpu(res, 0, index_ivf)

    assert not gpu_index_ivf.is_trained
    gpu_index_ivf.train(xb)        # add vectors to the index
    assert gpu_index_ivf.is_trained

    gpu_index_ivf.add(xb)          # add vectors to the index
    #  print(gpu_index_ivf.ntotal)

    k = K                         # we want to see k nearest neighbors

    # nlist - the number of cells (out of nlist) that are visited to perform a search
    gpu_index_ivf.nprobe = 1

    D, I = gpu_index_ivf.search(xq, k)  # actual search
    #  print(I[:5])                   # neighbors of the 5 first queries
    #  print(I[-5:])                  # neighbors of the 5 last queries
    return D, I

import tensorflow as tf

def calculate_recall(a,b):
    c = tf.sets.intersection(a,b)
    return c.values.shape[0]/(a.shape[0]*a.shape[1])

#  N_VERTS = [5000, 10000, 20000, 30000, 40000, 50000, 75000, 100000]
N_VERTS = [50000, 100000, 200000, 300000, 400000]
#  N_VERTS = [2000]
KS = [50, 100, 200, 500]
#  KS = [5]

using_index = using_IVF_index
outDataFile = "data/testing.txt"

#  using_index = using_flat_index
#  outDataFile = "data/perf_data_faiss_T4.txt"
#  using_index = using_IVF_index
#  outDataFile = "data/perf_data_faiss_IVF_T4.txt"

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

N_ITERS = 5
d = 4                           # dimension
np.random.seed(1234)             # make reproducible

import faiss                     # make faiss available

res = faiss.StandardGpuResources()  # use a single GPU

for K in KS:
    K = K+1
    for nb in N_VERTS:
        nq = nb                      # nb of queries
        np.random.seed(1234)             # make reproducible
        xb = np.random.random((nb, d)).astype('float32')
        xb[:, 0] += np.arange(nb) / 1000.
        xq = xb
        _, indx_tensor = using_index(d, xb, xq)
        t0 = time.time()
        for i in range(N_ITERS):
            using_index(d, xb, xq)
        exec_time = (time.time()-t0)/N_ITERS

        # Calculate recall:
        _, indx_tensor_ref = using_flat_index(d, xb, xq)
        recall = -1.0
        with tf.device('/CPU:0'):
            recall = calculate_recall(indx_tensor,indx_tensor_ref)

        f= open(outDataFile,"a+")
        print("%d\t%d\t%.3f\t%.2f\n" % (nb, K-1, exec_time, recall))
        f.write("%d\t%d\t%.3f\t%.2f\n" % (nb, K-1, exec_time, recall))
        f.close()

