import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
import tqdm


def test_img(net_g, datatest, args):
    net_g.eval()
    test_loss = 0
    sample_num = 0
    correct = 0  # number of correct predictions
    data_loader = DataLoader(datatest, batch_size=args.bs, shuffle=False)

    with tqdm.tqdm(total=len(data_loader)) as pbar:
        for idx, (data, target) in enumerate(data_loader):
            data, target = data.to(args.device), target.to(args.device)
            log_probs = net_g(data)

            # update test loss,sample number
            criterion = nn.CrossEntropyLoss()
            loss = criterion(log_probs, target)
            test_loss += loss.item()
            # test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
            sample_num += target.size(0)

            # get the index of the max log-probability
            _, y_pred = log_probs.max(1)
            # count number of correct predictions
            correct += y_pred.eq(target.view_as(y_pred)).long().cpu().sum()

            pbar.set_postfix_str('Tested:{}, Batch Loss: {:.4f}'.format(
                sample_num, loss.item()))
            pbar.update()

    test_loss /= sample_num     # average loss per sample
    accuracy = correct / sample_num

    return accuracy, test_loss
