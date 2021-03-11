# -*- coding: utf-8 -*-


import numpy as np
import torch
from scipy.special import comb

import utils
from utils import count_1, left_rotate, normalize, right_rotate


class GeneralPlanner(object):
    def __init__(self, scene_dim, device, inconsistency_state, action_set=None):
        self.scene_dim = scene_dim
        self.inconsistency_state = inconsistency_state
        if self.inconsistency_state:
            self.offset = 1
        else:
            self.offset = 0

        self.action_set = action_set

        # Define how the probability of a hidden relation is modeled
        # Will be used for torch.gather

        # constant
        self.constant_tri_valid = []
        self.constant_bi_valid = []
        for k in range(self.scene_dim):
            self.constant_tri_valid.append([k, k, k])
            self.constant_bi_valid.append([k, k])
        self.constant_tri_valid = torch.tensor(self.constant_tri_valid).to(device)
        self.constant_bi_valid = torch.tensor(self.constant_bi_valid).to(device)

        # progression one
        self.progression_one_tri_valid = []
        self.progression_one_bi_valid = []
        for k in range(self.scene_dim - self.offset - 2):
            self.progression_one_tri_valid.append([k, k + 1, k + 2])
            self.progression_one_bi_valid.append([k, k + 1])
        self.progression_one_tri_valid = torch.tensor(self.progression_one_tri_valid).to(device)
        self.progression_one_bi_valid = torch.tensor(self.progression_one_bi_valid).to(device)

        # progression two
        self.progression_two_tri_valid = []
        self.progression_two_bi_valid = []
        for k in range(self.scene_dim - self.offset - 4):
            self.progression_two_tri_valid.append([k, k + 2, k + 4])
            self.progression_two_bi_valid.append([k, k + 2])
        self.progression_two_tri_valid = torch.tensor(self.progression_two_tri_valid).to(device)
        self.progression_two_bi_valid = torch.tensor(self.progression_two_bi_valid).to(device)

        # progression mone
        self.progression_mone_tri_valid = []
        self.progression_mone_bi_valid = []
        for k in range(2, self.scene_dim - self.offset):
            self.progression_mone_tri_valid.append([k, k - 1, k - 2])
            self.progression_mone_bi_valid.append([k, k - 1])
        self.progression_mone_tri_valid = torch.tensor(self.progression_mone_tri_valid).to(device)
        self.progression_mone_bi_valid = torch.tensor(self.progression_mone_bi_valid).to(device)

        # progression mtwo
        self.progression_mtwo_tri_valid = []
        self.progression_mtwo_bi_valid = []
        for k in range(4, self.scene_dim - self.offset):
            self.progression_mtwo_tri_valid.append([k, k - 2, k - 4])
            self.progression_mtwo_bi_valid.append([k, k - 2])
        self.progression_mtwo_tri_valid = torch.tensor(self.progression_mtwo_tri_valid).to(device)
        self.progression_mtwo_bi_valid = torch.tensor(self.progression_mtwo_bi_valid).to(device)

        # arithmetic plus
        self.arithmetic_plus_tri_valid = []
        self.arithmetic_plus_bi_valid = []
        for i in range(self.scene_dim - self.offset):
            for j in range(self.scene_dim - self.offset):
                k = (i + 1) + (j + 1) - 1
                if k <= self.scene_dim - self.offset - 1:
                    self.arithmetic_plus_tri_valid.append([i, j, k])
                    self.arithmetic_plus_bi_valid.append([i, j])
        self.arithmetic_plus_tri_valid = torch.tensor(self.arithmetic_plus_tri_valid).to(device)
        self.arithmetic_plus_bi_valid = torch.tensor(self.arithmetic_plus_bi_valid).to(device)

        # arithmetic minus
        self.arithmetic_minus_tri_valid = []
        self.arithmetic_minus_bi_valid = []
        for i in range(self.scene_dim - self.offset):
            for j in range(self.scene_dim - self.offset):
                k = (i + 1) - (j + 1) - 1
                if k >= 0:
                    self.arithmetic_minus_tri_valid.append([i, j, k])
                    self.arithmetic_minus_bi_valid.append([i, j])
        self.arithmetic_minus_tri_valid = torch.tensor(self.arithmetic_minus_tri_valid).to(device)
        self.arithmetic_minus_bi_valid = torch.tensor(self.arithmetic_minus_bi_valid).to(device)

        # distribute three left
        self.distribute_three_left_valid = []
        for i in range(self.scene_dim - self.offset):
            for j in range(self.scene_dim - self.offset):
                for k in range(self.scene_dim - self.offset):
                    if i != j and j != k and i != k:
                        self.distribute_three_left_valid.append([i, j, k, j, k, i, k, i])
        self.distribute_three_left_valid = torch.tensor(self.distribute_three_left_valid).to(device)
    
        # distribute three right
        self.distribute_three_right_valid = []
        for i in range(self.scene_dim - self.offset):
            for j in range(self.scene_dim - self.offset):
                for k in range(self.scene_dim - self.offset):
                    if i != j and j != k and i != k:
                        self.distribute_three_right_valid.append([i, j, k, k, i, j, j, k])
        self.distribute_three_right_valid = torch.tensor(self.distribute_three_right_valid).to(device)

    def unit_action_prob(self, all_states, name):
        # all_states: log prob of shape (batch, 16, scene_dim)
        batch_size = all_states.shape[0]
        if name.startswith("distribute_three"):
            valid_indices = getattr(self, name + "_valid").unsqueeze(0).expand(batch_size, -1, -1).unsqueeze(-1).type(torch.long)
            valid_length = valid_indices.shape[1]
            expanded_states = all_states[:, :8, :].unsqueeze(1).expand(-1, valid_length, -1, -1)
            logprob = torch.sum(torch.gather(expanded_states, -1, valid_indices).squeeze(-1), dim=-1)
            return torch.sum(torch.exp(logprob), dim=-1), valid_length
        else:
            tri_valid_indices = getattr(self, name + "_tri_valid").unsqueeze(0).expand(batch_size, -1, -1).unsqueeze(-1).type(torch.long)
            bi_valid_indices = getattr(self, name + "_bi_valid").unsqueeze(0).expand(batch_size, -1, -1).unsqueeze(-1).type(torch.long)
            valid_length = tri_valid_indices.shape[1]
            # first row
            first_row = all_states[:, :3, :].unsqueeze(1).expand(-1, valid_length, -1, -1)
            first_row_logprob = utils.log(torch.sum(torch.exp(torch.sum(torch.gather(first_row, -1, tri_valid_indices).squeeze(-1), dim=-1)), dim=-1))
            # second row
            second_row = all_states[:, 3:6, :].unsqueeze(1).expand(-1, valid_length, -1, -1)
            second_row_logprob = utils.log(torch.sum(torch.exp(torch.sum(torch.gather(second_row, -1, tri_valid_indices).squeeze(-1), dim=-1)), dim=-1))
            # third row
            third_row = all_states[:, 6:8, :].unsqueeze(1).expand(-1, valid_length, -1, -1)
            third_row_logprob = utils.log(torch.sum(torch.exp(torch.sum(torch.gather(third_row, -1, bi_valid_indices).squeeze(-1), dim=-1)), dim=-1))
            return torch.exp((first_row_logprob + second_row_logprob + third_row_logprob)), (valid_length ** 3)
    
    def action_prob(self, all_states):
        prob = []
        for action_name in self.action_set:
            uni_action_prob, norm_constant = self.unit_action_prob(all_states, action_name)
            prob.append(uni_action_prob.unsqueeze(-1) / norm_constant)
        all_prob = torch.cat(prob, dim=-1)
        return normalize(all_prob)[0]


class PositionPlanner(GeneralPlanner):
    def __init__(self, scene_dim, device, action_set=["constant",
                                                      "progression_one", "progression_mone",
                                                      "arithmetic_plus", "arithmetic_minus", 
                                                      "distribute_three_left", "distribute_three_right"]):
        # super(PositionPlanner, self).__init__(scene_dim, device, False, action_set)
        self.scene_dim = scene_dim
        self.inconsistency_state = False
        if self.inconsistency_state:
            self.offset = 1
        else:
            self.offset = 0

        self.action_set = action_set
        self.num_slots = int(np.log2(scene_dim + 1))

        # constant
        self.constant_tri_valid = []
        self.constant_bi_valid = []
        for k in range(self.scene_dim):
            self.constant_tri_valid.append([k, k, k])
            self.constant_bi_valid.append([k, k])
        self.constant_tri_valid = torch.tensor(self.constant_tri_valid).to(device)
        self.constant_bi_valid = torch.tensor(self.constant_bi_valid).to(device)

        # progression one
        self.progression_one_tri_valid = []
        self.progression_one_bi_valid = []
        for k in range(self.scene_dim - self.offset):
            if count_1(k + 1) <= self.num_slots - 2:
                self.progression_one_tri_valid.append([k, right_rotate(k + 1, 1, self.num_slots) - 1, right_rotate(k + 1, 2, self.num_slots) - 1])
                self.progression_one_bi_valid.append([k, right_rotate(k + 1, 1, self.num_slots) - 1])
        self.progression_one_tri_valid = torch.tensor(self.progression_one_tri_valid).to(device)
        self.progression_one_bi_valid = torch.tensor(self.progression_one_bi_valid).to(device)

        # progression two
        self.progression_two_tri_valid = []
        self.progression_two_bi_valid = []
        for k in range(self.scene_dim - self.offset):
            if count_1(k + 1) <= self.num_slots - 4:
                self.progression_two_tri_valid.append([k, right_rotate(k + 1, 2, self.num_slots) - 1, right_rotate(k + 1, 4, self.num_slots) - 1])
                self.progression_two_bi_valid.append([k, right_rotate(k + 1, 2, self.num_slots) - 1])
        self.progression_two_tri_valid = torch.tensor(self.progression_two_tri_valid).to(device)
        self.progression_two_bi_valid = torch.tensor(self.progression_two_bi_valid).to(device)

        # progression mone
        self.progression_mone_tri_valid = []
        self.progression_mone_bi_valid = []
        for k in range(self.scene_dim - self.offset):
            if count_1(k + 1) <= self.num_slots - 2:
                self.progression_mone_tri_valid.append([k, left_rotate(k + 1, 1, self.num_slots) - 1, left_rotate(k + 1, 2, self.num_slots) - 1])
                self.progression_mone_bi_valid.append([k, left_rotate(k + 1, 1, self.num_slots) - 1])
        self.progression_mone_tri_valid = torch.tensor(self.progression_mone_tri_valid).to(device)
        self.progression_mone_bi_valid = torch.tensor(self.progression_mone_bi_valid).to(device)
    
        # progression mtwo
        self.progression_mtwo_tri_valid = []
        self.progression_mtwo_bi_valid = []
        for k in range(self.scene_dim - self.offset):
            if count_1(k + 1) <= self.num_slots - 4:
                self.progression_mtwo_tri_valid.append([k, left_rotate(k + 1, 2, self.num_slots) - 1, left_rotate(k + 1, 4, self.num_slots) - 1])
                self.progression_mtwo_bi_valid.append([k, left_rotate(k + 1, 2, self.num_slots) - 1])
        self.progression_mtwo_tri_valid = torch.tensor(self.progression_mtwo_tri_valid).to(device)
        self.progression_mtwo_bi_valid = torch.tensor(self.progression_mtwo_bi_valid).to(device)

        # arithmetic plus
        self.arithmetic_plus_tri_valid = []
        self.arithmetic_plus_bi_valid = []
        for i in range(self.scene_dim - self.offset):
            for j in range(self.scene_dim - self.offset):
                if i != j:
                    k = ((i + 1) | (j + 1)) - 1
                    self.arithmetic_plus_tri_valid.append([i, j, k])
                    self.arithmetic_plus_bi_valid.append([i, j])
        self.arithmetic_plus_tri_valid = torch.tensor(self.arithmetic_plus_tri_valid).to(device)
        self.arithmetic_plus_bi_valid = torch.tensor(self.arithmetic_plus_bi_valid).to(device)

        # arithmetic minus
        self.arithmetic_minus_tri_valid = []
        self.arithmetic_minus_bi_valid = []
        for i in range(self.scene_dim - self.offset):
            for j in range(self.scene_dim - self.offset):
                k = ((i + 1) & (~(j + 1))) - 1
                if k >= 0:
                    self.arithmetic_minus_tri_valid.append([i, j, k])
                    self.arithmetic_minus_bi_valid.append([i, j])
        self.arithmetic_minus_tri_valid = torch.tensor(self.arithmetic_minus_tri_valid).to(device)
        self.arithmetic_minus_bi_valid = torch.tensor(self.arithmetic_minus_bi_valid).to(device)
    
        # distribute three left and right
        self.distribute_three_left_valid = []
        self.distribute_three_right_valid = []
        for i in range(self.scene_dim - self.offset):
            for j in range(self.scene_dim - self.offset):
                for k in range(self.scene_dim - self.offset):
                    if (i != j and j != k and i != k) and (count_1(i) == count_1(j) and count_1(j) == count_1(k) and count_1(i) == count_1(k)):
                        self.distribute_three_left_valid.append([i, j, k, j, k, i, k, i])
                        self.distribute_three_right_valid.append([i, j, k, k, i, j, j, k])
        self.distribute_three_left_valid = torch.tensor(self.distribute_three_left_valid).to(device)
        self.distribute_three_right_valid = torch.tensor(self.distribute_three_right_valid).to(device)
        
    def action_prob(self, all_states):
        unnorm_prob = []
        prob = []
        for action_name in self.action_set:
            unit_action_prob, norm_constant = self.unit_action_prob(all_states, action_name)
            unnorm_prob.append(unit_action_prob.unsqueeze(-1))
            prob.append(unit_action_prob.unsqueeze(-1) / norm_constant)
        all_prob = torch.cat(prob, dim=-1)
        return normalize(all_prob)[0], unnorm_prob


class NumberPlanner(GeneralPlanner):
    def __init__(self, scene_dim, device, action_set=["progression_one", "progression_mone",
                                                      "arithmetic_plus", "arithmetic_minus",
                                                      "distribute_three_left", "distribute_three_right"]):
        super(NumberPlanner, self).__init__(scene_dim, device, False, action_set)
        self.normalization_num = [comb(self.scene_dim, i + 1) for i in range(self.scene_dim)]
        self.normalization_num = torch.tensor(self.normalization_num).to(device)

    def action_prob(self, all_states):
        unnorm_prob = []
        prob = []
        for action_name in self.action_set:
            unit_action_prob, norm_constant = self.unit_action_prob(all_states, action_name)
            unnorm_prob.append(unit_action_prob.unsqueeze(-1))
            prob.append(unit_action_prob.unsqueeze(-1) / norm_constant)
        all_prob = torch.cat(prob, dim=-1)
        return normalize(all_prob)[0], unnorm_prob


class TypePlanner(GeneralPlanner):
    def __init__(self, scene_dim, device, action_set=["constant",
                                                      "progression_one", "progression_mone", "progression_two", "progression_mtwo",
                                                      "distribute_three_left", "distribute_three_right"]):
        super(TypePlanner, self).__init__(scene_dim, device, True, action_set)


class SizePlanner(GeneralPlanner):
    def __init__(self, scene_dim, device, action_set=["constant",
                                                      "progression_one", "progression_mone", "progression_two", "progression_mtwo",
                                                      "arithmetic_plus", "arithmetic_minus", 
                                                      "distribute_three_left", "distribute_three_right"]):
        super(SizePlanner, self).__init__(scene_dim, device, True, action_set)
    

class ColorPlanner(GeneralPlanner):
    def __init__(self, scene_dim, device, action_set=["constant",
                                                      "progression_one", "progression_mone", "progression_two", "progression_mtwo",
                                                      "arithmetic_plus", "arithmetic_minus", 
                                                      "distribute_three_left", "distribute_three_right"]):
        super(ColorPlanner, self).__init__(scene_dim, device, True, action_set)        
        # arithmetic plus
        self.arithmetic_plus_tri_valid = []
        self.arithmetic_plus_bi_valid = []
        for i in range(self.scene_dim - self.offset):
            for j in range(self.scene_dim - self.offset):
                k = i + j
                if k <= self.scene_dim - self.offset - 1:
                    self.arithmetic_plus_tri_valid.append([i, j, k])
                    self.arithmetic_plus_bi_valid.append([i, j])
        self.arithmetic_plus_tri_valid = torch.tensor(self.arithmetic_plus_tri_valid).to(device)
        self.arithmetic_plus_bi_valid = torch.tensor(self.arithmetic_plus_bi_valid).to(device)

        # arithmetic minus
        self.arithmetic_minus_tri_valid = []
        self.arithmetic_minus_bi_valid = []
        for i in range(self.scene_dim - self.offset):
            for j in range(self.scene_dim - self.offset):
                k = i - j
                if k >= 0:
                    self.arithmetic_minus_tri_valid.append([i, j, k])
                    self.arithmetic_minus_bi_valid.append([i, j])
        self.arithmetic_minus_tri_valid = torch.tensor(self.arithmetic_minus_tri_valid).to(device)
        self.arithmetic_minus_bi_valid = torch.tensor(self.arithmetic_minus_bi_valid).to(device)
