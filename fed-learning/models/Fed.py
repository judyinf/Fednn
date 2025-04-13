"""
local weights aggregation
"""

import copy
import torch
import math


def FedAvg(w_locals, n_client):
    w_avg = {k: v * n_client[0] for k, v in w_locals[0].items()}
    for k in w_avg.keys():
        for i in range(1, len(w_locals)):
            w_avg[k] += w_locals[i][k] * n_client[i]
        w_avg[k] = torch.div(w_avg[k], sum(n_client))
    return w_avg


def fg(timestamp, t, a):
    if a == 1:
        return 1
    return math.pow(a, -t + timestamp)


def tw_fed(w_locals, w_glob, n_client, shallow_layer, timestamp_g, timestamp_s,
           t, asynchronous, flag, a, all_client, idx_client):
    w_avg = {}
    weights_g = [n_client[i] * fg(timestamp_g[i], t, a) for i in range(0, len(n_client))]
    weights_s = [n_client[i] * fg(timestamp_s[i], t, a) for i in range(0, len(n_client))]
    if not all_client:
        sum_weights_g = sum([weights_g[idx] for idx in idx_client])
        sum_weights_s = sum([weights_s[idx] for idx in idx_client])

    for key in w_locals[0].keys():
        if any(key.startswith(layer) for layer in shallow_layer):  # shallow layers
            if all_client:
                for i in range(0, len(n_client)):
                    if not i:
                        w_avg[key] = w_locals[i][key] * weights_g[i]
                    else:
                        w_avg[key] += w_locals[i][key] * weights_g[i]
                w_avg[key] = torch.div(w_avg[key], sum(weights_g))
            else:
                for i in range(0, len(idx_client)):
                    if not i:
                        w_avg[key] = w_locals[idx_client[i]][key] * weights_g[idx_client[i]]
                    else:
                        w_avg[key] += w_locals[idx_client[i]][key] * weights_g[idx_client[i]]
                w_avg[key] = torch.div(w_avg[key], sum_weights_g)

            w_glob[key] = copy.deepcopy(w_avg[key])

        elif (not asynchronous) or flag:  # update deep layers
            if all_client:
                for i in range(0, len(n_client)):
                    if not i:  # i=0
                        w_avg[key] = w_locals[i][key] * weights_s[i]
                    else:
                        w_avg[key] += w_locals[i][key] * weights_s[i]
                w_avg[key] = torch.div(w_avg[key], sum(weights_s))

            else:
                for i in range(0, len(idx_client)):
                    if not i:
                        w_avg[key] = w_locals[idx_client[i]][key] * weights_s[idx_client[i]]
                    else:
                        w_avg[key] += w_locals[idx_client[i]][key] * weights_s[idx_client[i]]
                w_avg[key] = torch.div(w_avg[key], sum_weights_s)

            w_glob[key] = copy.deepcopy(w_avg[key])

    return w_glob
