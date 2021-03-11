# -*- coding: utf-8 -*-


import numpy as np
import torch

from utils import count_1, left_rotate, normalize, right_rotate


class GeneralExecutor(object):
    def __init__(self, scene_dim, device, inconsistency_state, action_set=None):
        self.scene_dim = scene_dim
        self.scene_dim_2 = self.scene_dim ** 2
        self.inconsistency_state = inconsistency_state
        if self.inconsistency_state:
            self.offset = 1
        else:
            self.offset = 0
        self.action_set = action_set
        

        # Define how probability mass is transformed during execution
        # Will be used in matrix multiplication

        self.constant_trans = torch.zeros((self.scene_dim, self.scene_dim, self.scene_dim))
        for k in range(self.scene_dim):
            self.constant_trans[:, k, k] = 1.0
        self.constant_trans = self.constant_trans.view(self.scene_dim_2, self.scene_dim).to(device)
        
        self.progression_one_trans = torch.zeros((self.scene_dim, self.scene_dim, self.scene_dim))
        for k in range(self.scene_dim - self.offset - 2):
            self.progression_one_trans[:, k + 1, k + 2] = 1.0
        self.progression_one_trans = self.progression_one_trans.view(self.scene_dim_2, self.scene_dim).to(device)

        self.progression_mone_trans = torch.zeros((self.scene_dim, self.scene_dim, self.scene_dim))
        for k in range(2, self.scene_dim - self.offset):
            self.progression_mone_trans[:, k - 1, k - 2] = 1.0
        self.progression_mone_trans = self.progression_mone_trans.view(self.scene_dim_2, self.scene_dim).to(device)

        self.progression_two_trans = torch.zeros((self.scene_dim, self.scene_dim, self.scene_dim))
        for k in range(self.scene_dim - self.offset - 4):
            self.progression_two_trans[:, k + 2, k + 4] = 1.0
        self.progression_two_trans = self.progression_two_trans.view(self.scene_dim_2, self.scene_dim).to(device)

        self.progression_mtwo_trans = torch.zeros((self.scene_dim, self.scene_dim, self.scene_dim))
        for k in range(4, self.scene_dim - self.offset):
            self.progression_mtwo_trans[:, k - 2, k - 4] = 1.0
        self.progression_mtwo_trans = self.progression_mtwo_trans.view(self.scene_dim_2, self.scene_dim).to(device)

        self.arithmetic_plus_trans = torch.zeros((self.scene_dim, self.scene_dim, self.scene_dim))
        for i in range(self.scene_dim - self.offset):
            for j in range(self.scene_dim - self.offset):
                k = (i + 1) + (j + 1) - 1
                if k <= self.scene_dim - self.offset - 1:
                    self.arithmetic_plus_trans[i, j, k] = 1.0
        self.arithmetic_plus_trans = self.arithmetic_plus_trans.view(self.scene_dim_2, self.scene_dim).to(device)

        self.arithmetic_minus_trans = torch.zeros((self.scene_dim, self.scene_dim, self.scene_dim))
        for i in range(self.scene_dim - self.offset):
            for j in range(self.scene_dim - self.offset):
                k = (i + 1) - (j + 1) - 1
                if k >= 0:
                    self.arithmetic_minus_trans[i, j, k] = 1.0
        self.arithmetic_minus_trans = self.arithmetic_minus_trans.view(self.scene_dim_2, self.scene_dim).to(device)

        self.distribute_three_left_trans = torch.zeros((self.scene_dim, self.scene_dim, self.scene_dim))
        for k in range(self.scene_dim - self.offset):
            self.distribute_three_left_trans[:, k, k] = 1.0
        self.distribute_three_left_trans = self.distribute_three_left_trans.view(self.scene_dim_2, self.scene_dim).to(device)

        self.distribute_three_right_trans = torch.zeros((self.scene_dim, self.scene_dim, self.scene_dim))
        for k in range(self.scene_dim - self.offset):
            self.distribute_three_right_trans[k, :, k] = 1.0
        self.distribute_three_right_trans = self.distribute_three_right_trans.view(self.scene_dim_2, self.scene_dim).to(device)
    
    def gather_trans(self):
        # call this function at the end of __init__ in sub classes
        self.trans = torch.cat([getattr(self, trans_matrix_name + "_trans").unsqueeze(0) for trans_matrix_name in self.action_set], dim=0)

    def apply(self, all_states, action):
        is_distribute = (action >= len(self.action_set) - 2).unsqueeze(-1).unsqueeze(-1).type(torch.float)
        first_joint = torch.bmm(all_states[:, 0, :].unsqueeze(-1), all_states[:, 1, :].unsqueeze(1))
        third_joint = torch.bmm(all_states[:, 6, :].unsqueeze(-1), all_states[:, 7, :].unsqueeze(1))
        joint = is_distribute * first_joint + (1.0 - is_distribute) * third_joint
        trans = self.trans[action]
        pred = torch.bmm(joint.view(-1, 1, self.scene_dim_2), trans).squeeze(1)
        return normalize(pred)[0]
    

class PositionExecutor(GeneralExecutor):
    def __init__(self, scene_dim, device, action_set=["constant",
                                                      "progression_one", "progression_mone",
                                                      "arithmetic_plus", "arithmetic_minus", 
                                                      "distribute_three_left", "distribute_three_right"]):
        # super(PositionExecutor, self).__init__(scene_dim, device, False, action_set)
        self.scene_dim = scene_dim
        self.scene_dim_2 = self.scene_dim ** 2
        self.inconsistency_state = False
        if self.inconsistency_state:
            self.offset = 1
        else:
            self.offset = 0
        self.action_set = action_set
        
        # constant
        self.constant_trans = torch.zeros((self.scene_dim, self.scene_dim, self.scene_dim))
        for k in range(self.scene_dim):
            self.constant_trans[:, k, k] = 1.0
        self.constant_trans = self.constant_trans.view(self.scene_dim_2, self.scene_dim).to(device)

        self.num_slots = int(np.log2(scene_dim + 1))

        # progression one
        self.progression_one_trans = torch.zeros((self.scene_dim, self.scene_dim, self.scene_dim))
        for k in range(self.scene_dim - self.offset):
            if count_1(k + 1) <= self.num_slots - 2:
                self.progression_one_trans[:, right_rotate(k + 1, 1, self.num_slots) - 1, right_rotate(k + 1, 2, self.num_slots) - 1] = 1.0
        self.progression_one_trans = self.progression_one_trans.view(self.scene_dim_2, self.scene_dim).to(device)

        # progression two
        self.progression_two_trans = torch.zeros((self.scene_dim, self.scene_dim, self.scene_dim))
        for k in range(self.scene_dim - self.offset):
            if count_1(k + 1) <= self.num_slots - 4:
                self.progression_two_trans[:, right_rotate(k + 1, 2, self.num_slots) - 1, right_rotate(k + 1, 4, self.num_slots) - 1] = 1.0
        self.progression_two_trans = self.progression_two_trans.view(self.scene_dim_2, self.scene_dim).to(device)

        # progression mone
        self.progression_mone_trans = torch.zeros((self.scene_dim, self.scene_dim, self.scene_dim))
        for k in range(self.scene_dim - self.offset):
            if count_1(k + 1) <= self.num_slots - 2:
                self.progression_mone_trans[:, left_rotate(k + 1, 1, self.num_slots) - 1, left_rotate(k + 1, 2, self.num_slots) - 1] = 1.0
        self.progression_mone_trans = self.progression_mone_trans.view(self.scene_dim_2, self.scene_dim).to(device)
    
        # progression mtwo
        self.progression_mtwo_trans = torch.zeros((self.scene_dim, self.scene_dim, self.scene_dim))
        for k in range(self.scene_dim - self.offset):
            if count_1(k + 1) <= self.num_slots - 4:
                self.progression_mtwo_trans[:, left_rotate(k + 1, 2, self.num_slots) - 1, left_rotate(k + 1, 4, self.num_slots) - 1] = 1.0
        self.progression_mtwo_trans = self.progression_mtwo_trans.view(self.scene_dim_2, self.scene_dim).to(device)

        # arithmetic plus
        self.arithmetic_plus_trans = torch.zeros((self.scene_dim, self.scene_dim, self.scene_dim))
        for i in range(self.scene_dim - self.offset):
            for j in range(self.scene_dim - self.offset):
                if i != j:
                    k = ((i + 1) | (j + 1)) - 1
                    self.arithmetic_plus_trans[i, j, k] = 1.0
        self.arithmetic_plus_trans = self.arithmetic_plus_trans.view(self.scene_dim_2, self.scene_dim).to(device)

        # arithmetic minus
        self.arithmetic_minus_trans = torch.zeros((self.scene_dim, self.scene_dim, self.scene_dim))
        for i in range(self.scene_dim - self.offset):
            for j in range(self.scene_dim - self.offset):
                k = ((i + 1) & (~(j + 1))) - 1
                if k >= 0:
                    self.arithmetic_minus_trans[i, j, k] = 1.0
        self.arithmetic_minus_trans = self.arithmetic_minus_trans.view(self.scene_dim_2, self.scene_dim).to(device)

        # distribute three left
        self.distribute_three_left_trans = torch.zeros((self.scene_dim, self.scene_dim, self.scene_dim))
        for k in range(self.scene_dim - self.offset):
            self.distribute_three_left_trans[:, k, k] = 1.0
        self.distribute_three_left_trans = self.distribute_three_left_trans.view(self.scene_dim_2, self.scene_dim).to(device)

        # distribute three right
        self.distribute_three_right_trans = torch.zeros((self.scene_dim, self.scene_dim, self.scene_dim))
        for k in range(self.scene_dim - self.offset):
            self.distribute_three_right_trans[k, :, k] = 1.0
        self.distribute_three_right_trans = self.distribute_three_right_trans.view(self.scene_dim_2, self.scene_dim).to(device)

        self.gather_trans()


class NumberExecutor(GeneralExecutor):
    def __init__(self, scene_dim, device, action_set=["progression_one", "progression_mone",
                                                      "arithmetic_plus", "arithmetic_minus", 
                                                      "distribute_three_left", "distribute_three_right"]):
        super(NumberExecutor, self).__init__(scene_dim, device, False, action_set)
        self.gather_trans()


class TypeExecutor(GeneralExecutor):
    def __init__(self, scene_dim, device, action_set=["constant",
                                                      "progression_one", "progression_mone", "progression_two", "progression_mtwo",
                                                      "distribute_three_left", "distribute_three_right"]):
        super(TypeExecutor, self).__init__(scene_dim, device, True, action_set)
        self.gather_trans()


class SizeExecutor(GeneralExecutor):
    def __init__(self, scene_dim, device, action_set=["constant",
                                                      "progression_one", "progression_mone", "progression_two", "progression_mtwo",
                                                      "arithmetic_plus", "arithmetic_minus", 
                                                      "distribute_three_left", "distribute_three_right"]):
        super(SizeExecutor, self).__init__(scene_dim, device, True, action_set)
        self.gather_trans()


class ColorExecutor(GeneralExecutor):
    def __init__(self, scene_dim, device, action_set=["constant",
                                                      "progression_one", "progression_mone", "progression_two", "progression_mtwo",
                                                      "arithmetic_plus", "arithmetic_minus", 
                                                      "distribute_three_left", "distribute_three_right"]):
        super(ColorExecutor, self).__init__(scene_dim, device, True, action_set)
        self.arithmetic_plus_trans = torch.zeros((self.scene_dim, self.scene_dim, self.scene_dim))
        for i in range(self.scene_dim - self.offset):
            for j in range(self.scene_dim - self.offset):
                k = i + j
                if k <= self.scene_dim - self.offset - 1:
                    self.arithmetic_plus_trans[i, j, k] = 1.0
        self.arithmetic_plus_trans = self.arithmetic_plus_trans.view(self.scene_dim_2, self.scene_dim).to(device)

        self.arithmetic_minus_trans = torch.zeros((self.scene_dim, self.scene_dim, self.scene_dim))
        for i in range(self.scene_dim - self.offset):
            for j in range(self.scene_dim - self.offset):
                k = i - j
                if k >= 0:
                    self.arithmetic_minus_trans[i, j, k] = 1.0
        self.arithmetic_minus_trans = self.arithmetic_minus_trans.view(self.scene_dim_2, self.scene_dim).to(device)

        self.gather_trans()
