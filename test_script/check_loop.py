from options import args_parser

if __name__ == '__main__':
    flag = False
    args = args_parser()
    for iter in range(20):
        print("Epoch #{}".format(iter))
        if args.asynchronous and (not flag) and iter:
            print("Download only shallow layers")
        else:
            print("Download all layers")

        set1 =set(range(args.rounds - args.freq + 1, args.rounds)).union({0})
        if iter % args.rounds in set1:
            # {11, 12, 13, 14, 0}
            flag = True  # update all layers
            print("Update all layers")
        else:
            flag = False  # update only shallow layers
            print("Update only shallow layers")
