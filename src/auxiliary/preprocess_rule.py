# -*- code: utf-8 -*-


import glob
import os
import xml.etree.ElementTree as ET

import numpy as np
from tqdm import tqdm

# the rule/component idx: for configs with 2 components, run twice with different comp_idx
# left_right: 0 for left, 1 for right
# up_down: 0 for up, 1 for down
# in_out: 0 for out, 1 for in
# rule_idx = comp_idx = 0
# num_elements = 9 # 9 for distribute_nine and 4 for distribute_four

# other than 2x2 and 3x3
# comment out the part on number/position rule as there is no rules on number/position

# pos_num_rule_idx_map = None

# for distribute_four (2x2)
pos_num_rule_idx_map_four = {"Constant": 0,
                             "Progression_One_Pos": 1,
                             "Progression_Mone_Pos": 2,
                             "Arithmetic_Plus_Pos": 3,
                             "Arithmetic_Minus_Pos": 4,
                             "Distribute_Three_Left_Pos": 5,
                             "Distribute_Three_Right_Pos": 6,
                             "Progression_One_Num": 7,
                             "Progression_Mone_Num": 8,
                             "Arithmetic_Plus_Num": 9,
                             "Arithmetic_Minus_Num": 10,
                             "Distribute_Three_Left_Num": 11,
                             "Distribute_Three_Right_Num": 12}

# for distribute_nine (3x3)
pos_num_rule_idx_map_nine = {"Constant": 0,
                             "Progression_One_Pos": 1,
                             "Progression_Mone_Pos": 2,
                             "Progression_Two_Pos": 3,
                             "Progression_Mtwo_Pos": 4,
                             "Arithmetic_Plus_Pos": 5,
                             "Arithmetic_Minus_Pos": 6,
                             "Distribute_Three_Left_Pos": 7,
                             "Distribute_Three_Right_Pos": 8,
                             "Progression_One_Num": 9,
                             "Progression_Mone_Num": 10,
                             "Progression_Two_Num": 11,
                             "Progression_Mtwo_Num": 12,
                             "Arithmetic_Plus_Num": 13,
                             "Arithmetic_Minus_Num": 14,
                             "Distribute_Three_Left_Num": 15,
                             "Distribute_Three_Right_Num": 16}

# for other configs, pos_num_rule can only be 0 (Constant)
type_rule_idx_map = {"Constant": 0,
                     "Progression_One": 1,
                     "Progression_Mone": 2,
                     "Progression_Two": 3,
                     "Progression_Mtwo": 4,
                     "Distribute_Three_Left": 5,
                     "Distribute_Three_Right": 6}

size_rule_idx_map = {"Constant": 0,
                     "Progression_One": 1,
                     "Progression_Mone": 2,
                     "Progression_Two": 3,
                     "Progression_Mtwo": 4,
                     "Arithmetic_Plus": 5,
                     "Arithmetic_Minus": 6,
                     "Distribute_Three_Left": 7,
                     "Distribute_Three_Right": 8}

color_rule_idx_map = {"Constant": 0,
                      "Progression_One": 1,
                      "Progression_Mone": 2,
                      "Progression_Two": 3,
                      "Progression_Mtwo": 4,
                      "Arithmetic_Plus": 5,
                      "Arithmetic_Minus": 6,
                      "Distribute_Three_Left": 7,
                      "Distribute_Three_Right": 8}


def get_pos_num_rule(rule_idx, comp_idx, num_elements, pos_num_rule_idx_map, xml_panels, xml_rules):
    index_name = xml_rules[rule_idx][0].attrib["name"]
    attrib_name = xml_rules[rule_idx][0].attrib["attr"][:3]
    if index_name == "Progression":
        if attrib_name == "Num":
            first = int(xml_panels[0][0][comp_idx][0].attrib["Number"])
            second = int(xml_panels[1][0][comp_idx][0].attrib["Number"])
            if second == first + 1:
                index_name += "_One_Num"
            if second == first - 1:
                index_name += "_Mone_Num"
            if second == first + 2:
                index_name += "_Two_Num"
            if second == first - 2:
                index_name += "_Mtwo_Num"
        if attrib_name == "Pos":
            all_position = eval(xml_panels[0][0][comp_idx][0].attrib["Position"])
            first = []
            for entity in xml_panels[0][0][comp_idx][0]:
                first.append(all_position.index(eval(entity.attrib["bbox"])))
            second = []
            for entity in xml_panels[1][0][comp_idx][0]:
                second.append(all_position.index(eval(entity.attrib["bbox"])))
            third = []
            for entity in xml_panels[2][0][comp_idx][0]:
                third.append(all_position.index(eval(entity.attrib["bbox"])))
            fourth = []
            for entity in xml_panels[3][0][comp_idx][0]:
                fourth.append(all_position.index(eval(entity.attrib["bbox"])))
            fifth = []
            for entity in xml_panels[4][0][comp_idx][0]:
                fifth.append(all_position.index(eval(entity.attrib["bbox"])))
            sixth = []
            for entity in xml_panels[5][0][comp_idx][0]:
                sixth.append(all_position.index(eval(entity.attrib["bbox"])))
            seventh = []
            for entity in xml_panels[6][0][comp_idx][0]:
                seventh.append(all_position.index(eval(entity.attrib["bbox"])))
            eighth = []
            for entity in xml_panels[7][0][comp_idx][0]:
                eighth.append(all_position.index(eval(entity.attrib["bbox"])))
            if len(set(map(lambda index: (index + 1) % num_elements, first)) - set(second)) == 0 and \
               len(set(map(lambda index: (index + 1) % num_elements, second)) - set(third)) == 0 and \
               len(set(map(lambda index: (index + 1) % num_elements, fourth)) - set(fifth)) == 0 and \
               len(set(map(lambda index: (index + 1) % num_elements, fifth)) - set(sixth)) == 0 and \
               len(set(map(lambda index: (index + 1) % num_elements, seventh)) - set(eighth)) == 0:
                index_name += "_One_Pos"
            if len(set(map(lambda index: (index - 1) % num_elements, first)) - set(second)) == 0 and \
               len(set(map(lambda index: (index - 1) % num_elements, second)) - set(third)) == 0 and \
               len(set(map(lambda index: (index - 1) % num_elements, fourth)) - set(fifth)) == 0 and \
               len(set(map(lambda index: (index - 1) % num_elements, fifth)) - set(sixth)) == 0 and \
               len(set(map(lambda index: (index - 1) % num_elements, seventh)) - set(eighth)) == 0:
                index_name += "_Mone_Pos"
            if len(set(map(lambda index: (index + 2) % num_elements, first)) - set(second)) == 0 and \
               len(set(map(lambda index: (index + 2) % num_elements, second)) - set(third)) == 0 and \
               len(set(map(lambda index: (index + 2) % num_elements, fourth)) - set(fifth)) == 0 and \
               len(set(map(lambda index: (index + 2) % num_elements, fifth)) - set(sixth)) == 0 and \
               len(set(map(lambda index: (index + 2) % num_elements, seventh)) - set(eighth)) == 0:
                index_name += "_Two_Pos"
            if len(set(map(lambda index: (index - 2) % num_elements, first)) - set(second)) == 0 and \
               len(set(map(lambda index: (index - 2) % num_elements, second)) - set(third)) == 0 and \
               len(set(map(lambda index: (index - 2) % num_elements, fourth)) - set(fifth)) == 0 and \
               len(set(map(lambda index: (index - 2) % num_elements, fifth)) - set(sixth)) == 0 and \
               len(set(map(lambda index: (index - 2) % num_elements, seventh)) - set(eighth)) == 0:
                index_name += "_Mtwo_Pos"
            if index_name.endswith("_One_Pos_Mone_Pos"):
                if np.random.uniform() >= 0.5:
                    index_name = "Progression_One_Pos"
                else:
                    index_name = "Progression_Mone_Pos"
    if index_name == "Arithmetic":
        if attrib_name == "Num":
            first = int(xml_panels[0][0][comp_idx][0].attrib["Number"])
            second = int(xml_panels[1][0][comp_idx][0].attrib["Number"])
            third = int(xml_panels[2][0][comp_idx][0].attrib["Number"])
            if third == first + second + 1:
                index_name += "_Plus_Num"
            if third == first - second - 1:
                index_name += "_Minus_Num"
        if attrib_name == "Pos":
            all_position = eval(xml_panels[0][0][comp_idx][0].attrib["Position"])
            first = []
            for entity in xml_panels[0][0][comp_idx][0]:
                first.append(all_position.index(eval(entity.attrib["bbox"])))
            second = []
            for entity in xml_panels[1][0][comp_idx][0]:
                second.append(all_position.index(eval(entity.attrib["bbox"])))
            third = []
            for entity in xml_panels[2][0][comp_idx][0]:
                third.append(all_position.index(eval(entity.attrib["bbox"])))
            if set(third) == set(first).union(set(second)):
                index_name += "_Plus_Pos"
            if set(third) == set(first) - set(second):
                index_name += "_Minus_Pos"
    if index_name == "Distribute_Three":
        if attrib_name == "Num":
            first = int(xml_panels[0][0][comp_idx][0].attrib["Number"])
            second_left = int(xml_panels[5][0][comp_idx][0].attrib["Number"])
            second_right = int(xml_panels[4][0][comp_idx][0].attrib["Number"])
            if second_left == first:
                index_name += "_Left_Num"
            if second_right == first:
                index_name += "_Right_Num"
        if attrib_name == "Pos":
            all_position = eval(xml_panels[0][0][comp_idx][0].attrib["Position"])
            first = []
            for entity in xml_panels[0][0][comp_idx][0]:
                first.append(all_position.index(eval(entity.attrib["bbox"])))
            second_left = []
            for entity in xml_panels[5][0][comp_idx][0]:
                second_left.append(all_position.index(eval(entity.attrib["bbox"])))
            second_right = []
            for entity in xml_panels[4][0][comp_idx][0]:
                second_right.append(all_position.index(eval(entity.attrib["bbox"])))
            if set(second_left) == set(first):
                index_name += "_Left_Pos"
            if set(second_right) == set(first):
                index_name += "_Right_Pos"
    return pos_num_rule_idx_map[index_name]


def get_type_rule(rule_idx, comp_idx, num_elements, pos_num_rule_idx_map, xml_panels, xml_rules):
    index_name = xml_rules[rule_idx][1].attrib["name"]
    if index_name == "Progression":
        first = int(xml_panels[0][0][comp_idx][0][0].attrib["Type"])
        second = int(xml_panels[1][0][comp_idx][0][0].attrib["Type"])
        if second == first + 1:
            index_name += "_One"
        if second == first - 1:
            index_name += "_Mone"
        if second == first + 2:
            index_name += "_Two"
        if second == first - 2:
            index_name += "_Mtwo"
    if index_name == "Distribute_Three":
        first = int(xml_panels[0][0][comp_idx][0][0].attrib["Type"])
        second_left = int(xml_panels[5][0][comp_idx][0][0].attrib["Type"])
        second_right = int(xml_panels[4][0][comp_idx][0][0].attrib["Type"])
        if second_left == first:
            index_name += "_Left"
        if second_right == first:
            index_name += "_Right"
    return type_rule_idx_map[index_name]


def get_size_rule(rule_idx, comp_idx, num_elements, pos_num_rule_idx_map, xml_panels, xml_rules):
    index_name = xml_rules[rule_idx][2].attrib["name"]
    if index_name == "Progression":
        first = int(xml_panels[0][0][comp_idx][0][0].attrib["Size"])
        second = int(xml_panels[1][0][comp_idx][0][0].attrib["Size"])
        if second == first + 1:
            index_name += "_One"
        if second == first - 1:
            index_name += "_Mone"
        if second == first + 2:
            index_name += "_Two"
        if second == first - 2:
            index_name += "_Mtwo"
    if index_name == "Arithmetic":
        first = int(xml_panels[0][0][comp_idx][0][0].attrib["Size"])
        second = int(xml_panels[1][0][comp_idx][0][0].attrib["Size"])
        third = int(xml_panels[2][0][comp_idx][0][0].attrib["Size"])
        if third == first + second + 1:
            index_name += "_Plus"
        if third == first - second - 1:
            index_name += "_Minus"
    if index_name == "Distribute_Three":
        first = int(xml_panels[0][0][comp_idx][0][0].attrib["Size"])
        second_left = int(xml_panels[5][0][comp_idx][0][0].attrib["Size"])
        second_right = int(xml_panels[4][0][comp_idx][0][0].attrib["Size"])
        if second_left == first:
            index_name += "_Left"
        if second_right == first:
            index_name += "_Right"
    return size_rule_idx_map[index_name]


def get_color_rule(rule_idx, comp_idx, num_elements, pos_num_rule_idx_map, xml_panels, xml_rules):
    index_name = xml_rules[rule_idx][3].attrib["name"]
    if index_name == "Progression":
        first = int(xml_panels[0][0][comp_idx][0][0].attrib["Color"])
        second = int(xml_panels[1][0][comp_idx][0][0].attrib["Color"])
        if second == first + 1:
            index_name += "_One"
        if second == first - 1:
            index_name += "_Mone"
        if second == first + 2:
            index_name += "_Two"
        if second == first - 2:
            index_name += "_Mtwo"
    if index_name == "Arithmetic":
        first = int(xml_panels[0][0][comp_idx][0][0].attrib["Color"])
        second = int(xml_panels[1][0][comp_idx][0][0].attrib["Color"])
        third = int(xml_panels[2][0][comp_idx][0][0].attrib["Color"])
        fourth = int(xml_panels[3][0][comp_idx][0][0].attrib["Color"])
        fifth = int(xml_panels[4][0][comp_idx][0][0].attrib["Color"])
        sixth = int(xml_panels[5][0][comp_idx][0][0].attrib["Color"])
        if (third == first + second) and (sixth == fourth + fifth):
            index_name += "_Plus"
        if (third == first - second) and (sixth == fourth - fifth):
            index_name += "_Minus"
    if index_name == "Distribute_Three":
        first = int(xml_panels[0][0][comp_idx][0][0].attrib["Color"])
        second_left = int(xml_panels[5][0][comp_idx][0][0].attrib["Color"])
        second_right = int(xml_panels[4][0][comp_idx][0][0].attrib["Color"])
        if second_left == first:
            index_name += "_Left"
        if second_right == first:
            index_name += "_Right"
    return color_rule_idx_map[index_name]

    
def main():
    rule_idx = 0
    comp_idx = 0
    get_pos_num_rule_f = get_pos_num_rule

    # Only used by 2x2/3x3
    num_elements = 9
    pos_num_rule_idx_map = pos_num_rule_idx_map_four

    path = "/home/chizhang/Datasets/RAVEN-10000"
    configs = [
        ("center_single", 1),
        ("distribute_four", 1),
        ("distribute_nine", 1),
        ("left_center_single_right_center_single", 2),
        ("up_center_single_down_center_single", 2),
        ("in_center_single_out_center_single", 2),
        ("in_distribute_four_out_center_single", 2),
    ]
    for config, repeat_times in configs:
        if config == 'distribute_four':
            num_elements = 4
            pos_num_rule_idx_map = pos_num_rule_idx_map_four
            get_pos_num_rule_f = get_pos_num_rule
        elif config == 'distribute_nine':
            num_elements = 9
            pos_num_rule_idx_map = pos_num_rule_idx_map_nine
            get_pos_num_rule_f = get_pos_num_rule
        else:
            get_pos_num_rule_f = lambda *args: 0 # Constant

        files = glob.glob(os.path.join(path, config, "*.xml"))
        for i in range(repeat_times):
            rule_idx = i
            comp_idx = i
            print('Config: {}; Comp:{}'.format(config, comp_idx))
            for file in tqdm(files):
                xml_tree = ET.parse(file)
                xml_tree_root = xml_tree.getroot()
                xml_panels = xml_tree_root[0]
                xml_rules = xml_tree_root[1]
                new_file = file.replace(".xml", "_rule_comp{}.npz".format(comp_idx))
                args = [rule_idx, comp_idx, num_elements, pos_num_rule_idx_map, xml_panels, xml_rules]
                np.savez(new_file, pos_num_rule=get_pos_num_rule_f(*args),
                                type_rule=get_type_rule(*args),
                                size_rule=get_size_rule(*args),
                                color_rule=get_color_rule(*args))


if __name__ == '__main__':
    main()
