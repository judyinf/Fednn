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
from utils.Config import LSTMConfig, MnistConfig, HarConfig, CNNConfig
from utils.sampling import partition_iid, partition_non_iid
from preprocess.har import HARDataset
from models.Update import LocalUpdate, update_info_file
from models.Fed import tw_fed
from models.test import test_img

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
    output = os.path.join("output", "{}_{}_{}_{}_lr{}_lep{}".format(
        args.dataset, args.model, args.num_users, args.active_users, args.lr, args.local_ep))
    if args.iid:
        output += '_iid'
    else:
        output += '_niid'
    if args.asynchronous:
        output += "_async_{}".format(args.freq)
    if args.time_weighted:
        output += "_tw_{:.4f}".format(args.a)
    if args.all_clients:
        output += "_all_clients"
    os.makedirs(output, exist_ok=True)
    print(f"Output directory: {output}")
    os.makedirs(os.path.join(output, 'checkpoint', 'locals'), exist_ok=True)
    os.makedirs(os.path.join(output, 'checkpoint', 'central'), exist_ok=True)

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
    shallow_layer = {}
    if args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist().to(args.device)
        shallow_layer = CNNConfig.shallow_layer
    elif args.model == 'lstm' and args.dataset == 'har':
        input_size, hidden_sizes, output_size = LSTMConfig.input_size, LSTMConfig.hidden_sizes, LSTMConfig.output_size
        net_glob = LSTMHar(input_size, hidden_sizes, output_size).to(args.device)
        shallow_layer = LSTMConfig.shallow_layer
    else:
        exit('Error: unrecognized model')
    w_glob = net_glob.state_dict()

    # Load pretrained weights if available
    pretrained = args.pretrained
    if pretrained:
        w_locals, w_clients = [], []
        for i in range(args.num_users):
            locals_ckp_pth = os.path.join(output, 'checkpoint', 'central', f"checkpoint_{i:03d}.pt")
            clients_ckp_pth = os.path.join(output, 'checkpoint', 'clients', f"checkpoint_{i:03d}.pt")
            if os.path.exists(locals_ckp_pth):
                w_locals.append(torch.load(locals_ckp_pth, weights_only=True))
            else:
                w_locals.append(copy.deepcopy(w_glob))
            if os.path.exists(clients_ckp_pth):
                w_clients.append(torch.load(clients_ckp_pth, weights_only=True))
            else:
                w_clients.append(copy.deepcopy(w_glob))

        w_glob = torch.load(os.path.join(output, 'checkpoint', 'central', 'checkpoint_glob.pt'), weights_only=True)
        net_glob.load_state_dict(w_glob)
        net_clients = [copy.deepcopy(net_glob) for i in range(args.num_users)]
        for i in range(args.num_users):
            net_clients[i].load_state_dict(w_clients[i])
        with open(os.path.join(output, 'checkpoint', 'update_info.json'), 'r') as f:
            update_info = json.load(f)
            timestamp_g = update_info['timestamp_g']
            timestamp_s = update_info['timestamp_s']
            epoch_resume = update_info['epoch'] + 1
        print("Pretrained weights loaded. Resume training from epoch #{}".format(epoch_resume))

    else:  # initialize weights
        w_locals = [w_glob for i in range(args.num_users)]  # local model weights on central server
        w_clients = [w_glob for i in range(args.num_users)]  # local model weights on clients
        net_clients = [copy.deepcopy(net_glob) for i in range(args.num_users)]
        timestamp_g = [-1] * args.num_users  # general timestamp for shallow layers
        timestamp_s = [-1] * args.num_users  # specific timestamp for all layers
        epoch_resume = 0

    loss_train = []
    flag = True

    # training and testing
    for iter in range(epoch_resume, args.epochs):
        print("Epoch #{}".format(iter), flush=True)

        loss_locals = []
        m = args.active_users
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        # if args.asynchronous and (not flag):
        #     print("Download only shallow layers")
        # else:
        #     print("Download all layers")
        print("Download all layers")  # download all layers to local clients

        for idx in idxs_users:
            # download global weights to local clients
            # if args.asynchronous and (not flag):  # download only shallow layers
            #     for key in w_clients[idx].keys():
            #         if any(key.startswith(layer) for layer in shallow_layer):
            #             w_clients[idx][key] = copy.deepcopy(w_glob[key])  # copy shallow layer weights
            # else:
            #     w_clients[idx] = copy.deepcopy(w_glob)
            
            # download all layers to local clients
            w_clients[idx] = copy.deepcopy(w_glob)    
            net_clients[idx].load_state_dict(w_clients[idx])

        if iter % args.rounds in set(range(args.rounds - args.freq + 1, args.rounds)).union({0}):
            flag = True  # {11, 12, 13, 14, 0}
        else:
            flag = False

        if args.asynchronous and (not flag):
            print("Upload only shallow layers")  # upload only shallow layers to central server
        else:
            print("Upload all layers")  # upload all layers to central server

        for idx in idxs_users:
            # train local model
            print("Training client #{}".format(idx), flush=True)
            local = LocalUpdate(args=args, dataset=dataset_train,
                                idxs=dict_users[idx])  # idxs: index of data for each client

            # NB: w_clients[idx] is updated in place, both shallow and deep layers are updated
            w_clients[idx], loss = local.train(net_clients[idx].to(args.device))  # update local model weights

            # upload local weights to central server
            if args.asynchronous and (not flag):  # upload only shallow layers to central server
                timestamp_g[idx] = iter
                for key in w_locals[idx].keys():
                    if any(key.startswith(layer) for layer in shallow_layer):
                        w_locals[idx][key] = copy.deepcopy(w_clients[idx][key])
            else:  # upload all layers to central server
                timestamp_g[idx] = iter
                timestamp_s[idx] = iter
                w_locals[idx] = copy.deepcopy(w_clients[idx])

            # upload local loss to central server
            loss_locals.append(copy.deepcopy(loss))

            # save local model weights to local dir and central dir
            torch.save(w_clients[idx], os.path.join(output, 'checkpoint', 'locals',
                                                    f"checkpoint_{idx:03d}.pt"))
            torch.save(w_locals[idx], os.path.join(output, 'checkpoint', 'central',
                                                   f"checkpoint_{idx:03d}.pt"))
            update_info_file(json_path=os.path.join(output, 'checkpoint', 'update_info.json'),
                             central=False, idx=idx, n_timestamp_g=timestamp_g[idx], n_timestamp_s=timestamp_s[idx],
                             flag=flag, k=args.num_users, asynchronous=args.asynchronous)

        # weights aggregation on central server
        if args.time_weighted:
            a = args.a
        else:
            a = 1
        w_glob = tw_fed(w_locals, w_glob, n_client, shallow_layer,
               timestamp_g, timestamp_s, iter, args.asynchronous,
                        flag, a, args.all_clients, idxs_users)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)
        torch.save(w_glob, os.path.join(output, 'checkpoint', 'central', 'checkpoint_glob.pt'))
        update_info_file(json_path=os.path.join(output, 'checkpoint', 'update_info.json'),
                         central=True, epoch=iter)

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
            TensorWriter.add_scalar('Test/Accuracy', acc_test, iter)
            TensorWriter.add_scalar('Test/Loss', loss_test, iter)
            TensorWriter.add_scalar('Test/Train_Accuracy', acc_train, iter)
            TensorWriter.add_scalar('Test/Train_Loss', loss_test_tr, iter)
