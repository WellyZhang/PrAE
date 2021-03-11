# -*- coding: utf-8 -*-


import numpy as np
import torch

import utils


def calculate_acc(output, target):
    pred = output.data.max(-1)[1]
    correct = pred.eq(target.data).cpu().sum().numpy()
    return correct * 100.0 / np.prod(target.shape)


def calculate_correct(output, target):
    pred = output.data.max(-1)[1]
    correct = pred.eq(target.data).cpu().sum().numpy()
    return correct


def JSD(p, q):
    part1 = torch.sum(p * utils.log(2.0 * p) - p * utils.log(p + q), dim=-1)
    part2 = torch.sum(q * utils.log(2.0 * q) - q * utils.log(q + p), dim=-1)
    return 0.5 * part1 + 0.5 * part2


def JSD_unstable(p, q):
    m = (p + q) / 2
    return 0.5 * KL_divergence(p, m) + 0.5 * KL_divergence(q, m)


def KL_divergence(p, q):
    return torch.sum(p * utils.log(p / q), dim=-1)


def aux_loss(all_action_prob, all_action_rule):
    pos_num_action_prob, type_action_prob, size_action_prob, color_action_prob = all_action_prob
    pos_num_rule, type_rule, size_rule, color_rule = all_action_rule
    loss_pos_num = -utils.log(torch.gather(pos_num_action_prob, -1, pos_num_rule.unsqueeze(-1))).mean()
    loss_type = -utils.log(torch.gather(type_action_prob, -1, type_rule.unsqueeze(-1))).mean()
    loss_size = -utils.log(torch.gather(size_action_prob, -1, size_rule.unsqueeze(-1))).mean()
    loss_color = -utils.log(torch.gather(color_action_prob, -1, color_rule.unsqueeze(-1))).mean()
    loss = loss_pos_num + loss_type + loss_size + loss_color
    return loss
