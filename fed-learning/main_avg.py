import json
import os
import copy
import time

import numpy as np
from torchvision import datasets, transforms
import torch
from torch.utils.tensorboard import SummaryWriter
from utils.options import args_parser
from models.Nets import CNNMnist, LSTMHar
from utils.Config import LSTMConfig, MnistConfig, HarConfig
from utils.sampling import partition_iid, partition_non_iid
from preprocess.har import HARDataset
from models.Update import LocalUpdate
from models.test import test_img
from models.Fed import FedAvg

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    print(args)

    # Set seed
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # create output directory
    output = os.path.join("output_avg", "{}_{}_{}_{}".format(
        args.dataset, args.model, args.num_users, args.active_users))
    if args.iid:
        output += '_iid'
    else:
        output += '_niid'
    os.makedirs(output, exist_ok=True)
    print(f"Output directory: {output}")
    os.makedirs(os.path.join(output, 'checkpoint'), exist_ok=True)

    # -----Keep log-----#
    if args.keep_log:
        logtimestr = time.strftime("%m%d%H%M")
        boardpath = os.path.join(output, 'tensorboard_{}'.format(logtimestr))
        if not os.path.isdir(boardpath):
            os.makedirs(boardpath)
        TensorWriter = SummaryWriter(boardpath)

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = partition_iid(dataset_train, args.num_users)
        else:
            labels_list, Smin, Smax = MnistConfig.labels_list, MnistConfig.Smin, MnistConfig.Smax
            dict_users = partition_non_iid(dataset_train.targets, args.num_users, labels_list, Smin, Smax)
    elif args.dataset == 'har':
        trans_har = transforms.ToTensor()
        dataset_train = HARDataset(root="../data/UCI_HAR_Dataset/", train=True, transform=trans_har)
        dataset_test = HARDataset(root="../data/UCI_HAR_Dataset/", train=False, transform=trans_har)
        # sample users
        if args.iid:
            dict_users = partition_iid(dataset_train, args.num_users)
        else:
            labels_list, Smin, Smax = HarConfig.labels_list, HarConfig.Smin, HarConfig.Smax
            dict_users = partition_non_iid(dataset_train.labels, args.num_users, labels_list, Smin, Smax)

    else:
        exit('Error: unrecognized dataset')

    n_client = [len(dict_users[i]) for i in range(args.num_users)]  # client num list
    print("Clients sample number: ", n_client)

    # build model
    if args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist().to(args.device)
    elif args.model == 'lstm' and args.dataset == 'har':
        input_size, hidden_sizes, output_size = LSTMConfig.input_size, LSTMConfig.hidden_sizes, LSTMConfig.output_size
        net_glob = LSTMHar(input_size, hidden_sizes, output_size).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    w_glob = net_glob.state_dict()

    # Load pretrained weights if available
    pretrained = args.pretrained
    if pretrained:
        w_glob = torch.load(os.path.join(output, 'checkpoint', 'checkpoint_glob.pt'), weights_only=True)
        net_glob.load_state_dict(w_glob)
        with open(os.path.join(output, 'checkpoint', 'update_info.json'), 'r') as f:
            update_info = json.load(f)
            epoch_resume = update_info['epoch'] + 1
        print("Pretrained weights loaded. Resume training from epoch #{}".format(epoch_resume))
    else:  # initialize weights
        epoch_resume = 0

    loss_train = []
    for iter in range(epoch_resume, args.epochs):
        print("Epoch #{}".format(iter), flush=True)
        loss_locals = []
        w_locals = []
        n_locals = []
        m = args.active_users
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        for idx in idxs_users:
            # train local model
            print("Training client #{}".format(idx), flush=True)
            local = LocalUpdate(args=args, dataset=dataset_train,
                                idxs=dict_users[idx])  # idxs: index of data for each client

            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))  # update local model weights
            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
            n_locals.append(n_client[idx])

        w_glob = FedAvg(w_locals, n_locals)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)
        torch.save(w_glob, os.path.join(output, 'checkpoint', 'checkpoint_glob.pt'))
        json_path = os.path.join(output, 'checkpoint', 'update_info.json')
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            data = {'epoch': 0}
        with open(json_path, 'w') as g:
            data['epoch'] = iter
            json.dump(data, g)

        # log loss_train
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)
        if args.keep_log:
            TensorWriter.add_scalar('Train/Loss_Avg', loss_avg, iter)

        # testing
        net_glob.eval()
        acc_train, loss_test_tr = test_img(net_glob, dataset_train, args)
        acc_test, loss_test = test_img(net_glob, dataset_test, args)
        print("Training accuracy: {:.4f},Training loss: {:.4f}".format(acc_train, loss_test_tr))
        print("Testing accuracy: {:.4f},Testing loss: {:.4f}".format(acc_test, loss_test))
        if args.keep_log:
            TensorWriter.add_scalar('Test/Accuracy_te', acc_test, iter)
            TensorWriter.add_scalar('Test/Loss_te', loss_test, iter)
            TensorWriter.add_scalar('Test/Accuracy_tr', acc_train, iter)
            TensorWriter.add_scalar('Test/Loss_tr', loss_test_tr, iter)
