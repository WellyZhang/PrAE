# coding: utf-8 -*-


import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import trange

import criteria
import network
from const import DIM_COLOR, DIM_EXIST, DIM_SIZE, DIM_TYPE
from dataset import Dataset
from env import get_env


def check_paths(args):
    try:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
        new_log_dir = os.path.join(args.log_dir, time.ctime().replace(" ", "-"))
        args.log_dir = new_log_dir
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
        if not os.path.exists(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir)
    except OSError as e:
        print(e)
        sys.exit(1)


def train(args, env, device):
    def train_epoch(epoch, steps):
        model.train()
        xe_avg = 0.0
        counter = 0
        train_loader_iter = iter(train_loader)
        for _ in trange(len(train_loader_iter)):
            steps += 1
            counter += 1
            images, targets, all_action_rule = next(train_loader_iter)
            images = images.to(device)
            targets = targets.to(device)
            all_action_rule_device = []
            for action_rule in all_action_rule:
                action_rule = action_rule.to(device)
                all_action_rule_device.append(action_rule)
            model_output = model(images)
            scene_prob, scene_logprob = env.prepare(model_output)
            action, action_logprob, all_action_prob = env.action(scene_logprob)
            pred = env.step(scene_prob, action)
            loss, scores, xe_loss_item = env.loss(action[0], action_logprob, pred, scene_prob, targets, error_function)
            aux_loss = criteria.aux_loss(all_action_prob, all_action_rule_device) 
            final_loss = loss + args.aux * aux_loss
            acc = criteria.calculate_acc(scores, targets)
            xe_avg += xe_loss_item
            optimizer.zero_grad()
            final_loss.backward()
            optimizer.step()
        print("Epoch {}, Total Iter: {}, Train Avg XE: {:.6f}".format(epoch, counter, xe_avg / float(counter)))

        return steps

    def validate_epoch(epoch, steps):
        model.eval()
        loss_avg = 0.0
        acc_avg = 0.0
        xe_avg = 0.0
        counter = 0
        val_loader_iter = iter(val_loader)
        for _ in trange(len(val_loader_iter)):
            counter += 1
            images, targets, _ = next(val_loader_iter)
            images = images.to(device)
            targets = targets.to(device)
            model_output = model(images)
            scene_prob, scene_logprob = env.prepare(model_output)
            action, action_logprob, _ = env.action(scene_logprob, sample=False)
            pred = env.step(scene_prob, action)
            loss, scores, xe_loss_item = env.loss(action[0], action_logprob, pred, scene_prob, targets, error_function)
            loss_avg += loss.item()
            acc = criteria.calculate_acc(scores, targets)
            acc_avg += acc.item()
            xe_avg += xe_loss_item
        print("Epoch {}, Valid Avg XE: {:.6f}, Valid Avg Acc: {:.4f}".format(epoch, xe_avg / float(counter), acc_avg / float(counter)))
    
    def test_epoch(epoch, steps):
        model.eval()
        loss_avg = 0.0
        acc_avg = 0.0
        xe_avg = 0.0
        counter = 0
        test_loader_iter = iter(test_loader)
        for _ in trange(len(test_loader_iter)):
            counter += 1
            images, targets, _ = next(test_loader_iter)
            images = images.to(device)
            targets = targets.to(device)
            model_output = model(images)
            scene_prob, scene_logprob = env.prepare(model_output)
            action, action_logprob, _ = env.action(scene_logprob, sample=False)
            pred = env.step(scene_prob, action)
            loss, scores, xe_loss_item = env.loss(action[0], action_logprob, pred, scene_prob, targets, error_function)
            loss_avg += loss.item()
            acc = criteria.calculate_acc(scores, targets)
            acc_avg += acc.item()
            xe_avg += xe_loss_item
        print("Epoch {}, Test  Avg XE: {:.6f}, Test  Avg Acc: {:.4f}".format(epoch, xe_avg / float(counter), acc_avg / float(counter)))

    # set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # init model
    model = network.Perception(DIM_EXIST, DIM_TYPE, DIM_SIZE, DIM_COLOR)
    if args.resume is not None:
        model.load_state_dict(torch.load(args.resume))
    # for the old traning strategy of curricular training
    # it is likely not necessary in this new version
    # for param in model.exist_model.parameters():
    #     param.requires_grad = False
    # for param in model.type_model.parameters():
    #     param.requires_grad = False
    # for param in model.size_model.parameters():
    #     param.requires_grad = False
    # for param in model.color_model.parameters():
    #     param.requires_grad = False
    model.to(device)
    error_function = getattr(criteria, args.error)
    optimizer = optim.Adam([param for param in model.parameters() if param.requires_grad], args.lr, weight_decay=args.weight_decay)

    # dataset loader
    train_set = Dataset(args.dataset, "train", args.img_size, args.config)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_set = Dataset(args.dataset, "val", args.img_size, args.config, test=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=args.num_workers)
    test_set = Dataset(args.dataset, "test", args.img_size, args.config, test=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.num_workers)
    
    total_steps = 0

    # training loop starts
    for epoch in range(args.epochs):
        total_steps = train_epoch(epoch, total_steps)
        with torch.no_grad():
            validate_epoch(epoch, total_steps)
            test_epoch(epoch, total_steps)
        
        # save checkpoint    
        model.eval().cpu()
        ckpt_model_name = "epoch_{}_batch_{}_seed_{}_config_{}_img_{}_lr_{}_l2_{}_error_{}_aux_{}.pth".format(
            epoch, 
            args.batch_size, 
            args.seed,
            args.config,
            args.img_size,
            args.lr, 
            args.weight_decay,
            args.error,
            args.aux)
        ckpt_file_path = os.path.join(args.checkpoint_dir, ckpt_model_name)
        torch.save(model.state_dict(), ckpt_file_path)
        model.to(device)
        
    # save final model
    model.eval().cpu()
    save_model_name = "Final_epoch_{}_batch_{}_seed_{}_config_{}_img_{}_lr_{}_l2_{}_error_{}_aux_{}.pth".format(
        epoch, 
        args.batch_size, 
        args.seed,
        args.config,
        args.img_size,
        args.lr,
        args.weight_decay,
        args.error,
        args.aux)
    save_file_path = os.path.join(args.save_dir, save_model_name)
    torch.save(model.state_dict(), save_file_path)

    print("Done. Model saved.")


def test(args, env, device):
    def test_epoch():
        model.eval()
        loss_avg = 0.0
        correct_avg = 0.0
        xe_avg = 0.0
        test_loader_iter = iter(test_loader)
        for _ in trange(len(test_loader_iter)):
            images, targets, _ = next(test_loader_iter)
            images = images.to(device)
            targets = targets.to(device)
            model_output = model(images)
            scene_prob, scene_logprob = env.prepare(model_output)
            action, action_logprob, _ = env.action(scene_logprob, sample=False)
            pred = env.step(scene_prob, action)
            loss, scores, xe_loss_item = env.loss(action[0], action_logprob, pred, scene_prob, targets, error_function)
            loss_avg += loss.item() * images.shape[0]
            correct_num = criteria.calculate_correct(scores, targets)
            correct_avg += correct_num.item()
            xe_avg += xe_loss_item * images.shape[0]
        # print "Test Avg Loss: {:.4f}".format(loss_avg / float(test_set_size))
        print("Test Avg Acc: {:.4f}".format(correct_avg / float(test_set_size)))
        print("Test Avg XE: {:.4f}".format(xe_avg / float(test_set_size)))

    model = network.Perception(DIM_EXIST, DIM_TYPE, DIM_SIZE, DIM_COLOR)
    model.load_state_dict(torch.load(args.model_path))
    model.to(device)
    error_function = getattr(criteria, args.error)
    test_set = Dataset(args.dataset, "test", args.img_size, args.config, test=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.num_workers)
    test_set_size = len(test_set)
    print("Evaluating on {}".format(args.config))
    with torch.no_grad():
        test_epoch()


def main(): 
    main_arg_parser = argparse.ArgumentParser(description="The PrAE Learner")
    subparsers = main_arg_parser.add_subparsers(title="subcommands", dest="subcommand")
    
    train_arg_parser = subparsers.add_parser("train", help="parser for training")
    train_arg_parser.add_argument("--epochs", type=int, default=200,
                                  help="the number of training epochs")
    train_arg_parser.add_argument("--batch-size", type=int, default=32,
                                  help="size of batch")
    train_arg_parser.add_argument("--seed", type=int, default=1234,
                                  help="random number seed")
    train_arg_parser.add_argument("--device", type=int, default=0,
                                  help="device index for GPU; if GPU unavailable, leave it as default")
    train_arg_parser.add_argument("--dataset", type=str, default="/home/chizhang/Datasets/RAVEN-10000/",
                                  help="dataset path")
    train_arg_parser.add_argument("--config", type=str, default="distribute_four",
                                  help="the configuration used for training")
    train_arg_parser.add_argument("--checkpoint-dir", type=str, default="./runs/ckpt/",
                                  help="checkpoint save path")
    train_arg_parser.add_argument("--save-dir", type=str, default="./runs/save/",
                                  help="final model save path")
    train_arg_parser.add_argument("--log-dir", type=str, default="./runs/log/",
                                  help="log save path")
    train_arg_parser.add_argument("--img-size", type=int, default=32,
                                  help="image region size for training")
    train_arg_parser.add_argument("--lr", type=float, default=0.95e-4,
                                  help="learning rate")
    train_arg_parser.add_argument("--weight-decay", type=float, default=0.0,
                                  help="weight decay of optimizer, same as l2 reg")
    train_arg_parser.add_argument("--error", type=str, default="JSD",
                                  help="error used to measure difference between distributions")
    train_arg_parser.add_argument("--num-workers", type=int, default=2,
                                  help="number of workers for data loader")
    train_arg_parser.add_argument("--resume", type=str, default=None,
                                  help="resume from a initialized model")
    train_arg_parser.add_argument("--aux", type=float, default=1.0,
                                  help="weight of auxiliary training")

    test_arg_parser = subparsers.add_parser("test", help="parser for testing")
    test_arg_parser.add_argument("--batch-size", type=int, default=32,
                                 help="size of batch")
    test_arg_parser.add_argument("--device", type=int, default=0,
                                 help="device index for GPU; if GPU unavailable, leave it as default")
    test_arg_parser.add_argument("--dataset", type=str, default="/home/chizhang/Datasets/RAVEN-10000",
                                 help="dataset path")
    test_arg_parser.add_argument("--config", type=str, default="distribute_four",
                                 help="the configuration used for testing")
    test_arg_parser.add_argument("--model-path", type=str, required=True,
                                 help="path to a trained model")
    test_arg_parser.add_argument("--img-size", type=int, default=32,
                                 help="image region size for training")
    test_arg_parser.add_argument("--error", type=str, default="JSD",
                                 help="error used to measure difference betweeen distributions")
    test_arg_parser.add_argument("--shuffle", type=int, default=0,
                                 help="whether to shuffle the dataset")
    test_arg_parser.add_argument("--num-workers", type=int, default=2,
                                 help="number of workers for data loader")

    args = main_arg_parser.parse_args()
    args.cuda = torch.cuda.is_available()
    device = torch.device("cpu")
    # device = torch.device("cuda:{}".format(args.device) if args.cuda else "cpu")

    env = get_env(args.config, device)
    if env is None:
        print("ERROR: Unsupported environment")
        sys.exit(1)

    if args.subcommand is None:
        print("ERROR: Specify train or test")
        sys.exit(1)

    if args.subcommand == "train":
        check_paths(args)
        train(args, env, device)
    elif args.subcommand == "test":
        test(args, env, device)
    else:
        print("ERROR: Unknown subcommand")
        sys.exit(1)


if __name__ == "__main__":
    main()
