import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import json


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        data, label = self.dataset[self.idxs[item]]
        return data, label


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

    def train(self, net):
        net.train()
        # train and update
        # lr = self.args.lr; momentum = self.args.momentum # SGD
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        epoch_l2 = []

        for iter in range(self.args.local_ep):
            batch_loss = []
            sample_num = 0
            # with tqdm.tqdm(total=len(self.ldr_train)) as pbar:
            for batch_idx, (data, labels) in enumerate(self.ldr_train):
                data, labels = data.to(self.args.device), labels.to(self.args.device)

                net.zero_grad()
                log_probs = net(data)
                loss = self.loss_func(log_probs, labels)
                loss.backward()

                optimizer.step()

                sample_num += labels.size(0)
                batch_loss.append(loss.item())

                # pbar.set_postfix_str('Local Epoch:{},Trained:{},Batch Loss:{:.4f}'.format(
                #     iter, sample_num, loss.item()))
                # pbar.update()
            
            epoch_loss.append(sum(batch_loss) / sample_num)  # average loss per sample

        print(f"Averaged L2 norm: {sum(epoch_l2) / len(epoch_l2)}")
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


def update_info_file(json_path, central=False, epoch=0, idx=0, n_timestamp_g=0, n_timestamp_s=0,
                     k=20, flag=False, asynchronous=False):
    # Load or initialize the JSON data
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        # Initialize the file structure if it doesn't exist
        data = {'local_update': {}, 'epoch': 0,
                'timestamp_g': [-1] * k, 'timestamp_s': [-1] * k}

    if central:
        data['epoch'] = epoch  # update epoch
    else:
        client_id = f"{int(idx):03d}"
        # Check if the client_id exists, if not initialize it
        if client_id not in data['local_update'].keys():
            data['local_update'][client_id] = {
                'update_g': [],
                'update_s': [],
                'time_stamp_g': -1,
                'time_stamp_s': -1
            }
        client_data = data['local_update'][client_id]
        client_data['update_g'].append(n_timestamp_g)
        client_data['time_stamp_g'] = n_timestamp_g
        data['timestamp_g'][idx] = n_timestamp_g
        if (asynchronous and flag) or (not asynchronous):  # update all layers
            client_data['update_s'].append(n_timestamp_s)
            client_data['time_stamp_s'] = n_timestamp_s
            data['timestamp_s'][idx] = n_timestamp_s

    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)
