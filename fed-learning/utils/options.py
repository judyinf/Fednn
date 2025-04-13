import argparse
import math


def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=50, help="rounds of training")
    # K,m
    parser.add_argument('--num_users', type=int, default=20, help="number of users: K")
    parser.add_argument('--active_users', type=int, default=2, help="the num of active clients: m")
    # freq,a
    parser.add_argument('--asynchronous', action='store_true', help='asynchronous training')
    parser.add_argument('--rounds', type=int, default=15, help='communication rounds in a loop')
    parser.add_argument('--freq', type=int, default=5, help='frequency of deep layer update')
    parser.add_argument('--time_weighted', action='store_true', help='time weighted aggregation')
    parser.add_argument('--a', type=float, default=math.e/2, help='a in the power law distribution')

    # model arguments
    parser.add_argument('--model', type=str, default='cnn', help='model name')
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")

    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')

    # other arguments
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 1)')
    parser.add_argument('--keep_log', action='store_true', help='keep log')
    parser.add_argument('--pretrained', action='store_true', help='use pretrained model')
    parser.add_argument('--all_clients', action='store_true', help='aggregation over all clients')

    args = parser.parse_args()

    return args
