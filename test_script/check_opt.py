
import numpy as np
from options import args_parser
if __name__ == '__main__':
    args = args_parser()
    print(args)
    np.random.seed(args.seed)
    flag = False
    iter = 15
    if iter % args.rounds in set(range(args.rounds - args.freq + 1, args.rounds)).union({0}):  # {11, 12, 13, 14, 0}
        flag = True  # update all layers
    print(flag)
