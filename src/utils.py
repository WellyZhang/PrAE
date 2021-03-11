# -*- coding: utf-8 -*-


import torch

from const import LOG_EPSILON, NORM_SCALE


def normalize(unnorm_prob, dim=-1):
    unnorm_prob = unnorm_prob * NORM_SCALE
    sum_dim = torch.sum(unnorm_prob, dim=dim, keepdim=True)
    norm_prob = unnorm_prob / sum_dim
    return norm_prob, sum_dim
    

def to_n_bit_string(n, number):
    format_string = "{" + "0:0{}b".format(n) + "}"
    return format_string.format(number)


def left_rotate(number, steps, num_bits):
    offset = steps % num_bits
    index = ((number << offset) | (number >> (num_bits - offset))) & (2 ** num_bits - 1)
    return index


def right_rotate(number, steps, num_bits):
    offset = steps % num_bits
    index = ((number >> offset) | (number << (num_bits - offset))) & (2 ** num_bits - 1)
    return index


def count_1(n):
    return bin(n).count("1")


def sample_action(prob, sample=True):
    if sample:
        action = torch.distributions.Categorical(prob).sample()
    else:
        action = torch.argmax(prob, dim=-1)
    logprob = torch.log(torch.gather(prob, -1, action.unsqueeze(-1))).squeeze(-1)
    return action, logprob


def log(x):
    return torch.log(x + LOG_EPSILON)
