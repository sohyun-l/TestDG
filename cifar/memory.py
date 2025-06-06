import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class Memory(nn.Module):
    def __init__(self, queue_size, channel, gpu_idx, alpha=0.1):
        super(Memory, self).__init__()
        self.queue_size = queue_size
        self.mem = torch.zeros((self.queue_size, channel)).cuda(gpu_idx)
        self.labels = torch.zeros(self.queue_size, dtype=torch.long).cuda(gpu_idx)
        self.domains = torch.zeros(self.queue_size, dtype=torch.long).cuda(gpu_idx)
        self.gpu_idx = gpu_idx
        self.ptr = 0
        self.alpha = alpha

        self.is_full = False

        self.use_norm = False
        self.channel = channel

    def clear(self):
        self.is_full = False
        self.ptr = 0

    def enqueue(self, feats, domains=None, labels=None):
        num_element = feats.size(0)

        if self.ptr + num_element >= self.queue_size:
            pre_idx = self.queue_size - self.ptr
            self.mem[self.ptr :,] = feats[:pre_idx,]
            self.mem[: num_element - pre_idx,] = feats[pre_idx:,]

            if domains is not None:
                self.domains[self.ptr :,] = domains[:pre_idx]
                self.domains[: num_element - pre_idx] = domains[pre_idx:]

            if labels is not None:
                self.labels[self.ptr :,] = labels[:pre_idx]
                self.labels[: num_element - pre_idx] = labels[pre_idx:]

            self.ptr = num_element - pre_idx

            self.is_full = True
        else:
            self.mem[self.ptr : self.ptr + num_element] = feats

            if domains is not None:
                self.domains[self.ptr : self.ptr + num_element] = domains
            if labels is not None:
                self.labels[self.ptr : self.ptr + num_element] = labels

            self.ptr += num_element

        return

    def get_random_mem(self, num=32, return_idx=False):
        batch_idx = torch.randperm(
            self.queue_size if self.is_full else self.ptr, device=self.gpu_idx
        )
        batch_idx = batch_idx[:num]

        value = self.mem[batch_idx, :]

        if return_idx:
            return value, batch_idx

        return value

    def get_random_mem_from_last_generated(self, last_generated_num, num=32):
        batch_idx = torch.randperm(last_generated_num, device=self.gpu_idx)[:num]
        batch_idx = (
            batch_idx + (self.ptr - last_generated_num + self.queue_size)
        ) % self.queue_size

        value = self.mem[batch_idx, :]

        return value

    def get_all_mem_from_last_generated(self, last_generated_num):
        batch_idx = torch.arange(last_generated_num, device=self.gpu_idx)
        batch_idx = (
            batch_idx + (self.ptr - last_generated_num + self.queue_size)
        ) % self.queue_size

        value = self.mem[batch_idx, :]

        return value

    def get_mem(self):
        if self.is_full:
            return self.mem
        else:
            return self.mem[: self.ptr]

    def get_mem_wo_drop(self, generated_num):
        if self.ptr + generated_num >= self.queue_size:
            return self.mem[(self.ptr + generated_num) % self.queue_size : self.ptr]
        else:
            index = torch.arange(0, self.ptr)
            if self.is_full:
                index = torch.cat(
                    [index, torch.arange(self.ptr + generated_num, self.queue_size)],
                    dim=0,
                )
            return self.mem[index, :]

    def get_num(self):
        if self.is_full:
            return self.queue_size
        else:
            return self.ptr
