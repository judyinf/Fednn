import numpy as np


def partition_iid(data, num_clients):
    """
    Sample I.I.D. client data
    :param data:
    :param num_clients:
    :return: dict of image index
    The dict_users dictionary has the following structure:
    Key: An integer representing the user ID.
    Value: A list of indices of the data samples that are assigned to the user.
    """
    np.random.seed(0)
    num_items = int(len(data)/num_clients)
    dict_users, all_idxs = {}, [i for i in range(len(data))]
    for i in range(num_clients):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
        dict_users[i] = list(dict_users[i])
    return dict_users


def partition_non_iid(labels, num_clients=20, labels_list=None, Smin=250, Smax=500):
    """
    Partitions the dataset into non-IID and unbalanced local datasets for clients.

    Args:
        labels (torch.Tensor): A 1D tensor containing the labels of the dataset.
        num_clients (int): The number of clients to partition the dataset into.
        labels_list (list): The set of possible labels.
        Smin (int): Minimum number of samples for a client.
        Smax (int): Maximum number of samples for a client.

    Returns:
        dict_users: A dictionary containing the indices of the samples assigned to each client.
    """
    np.random.seed(0)

    if labels_list is None:
        labels_list = [0, 1, 2, 3, 4, 5]
    dict_users = {}  # Initialize dictionary to hold data for each client

    for client_id in range(num_clients):
        # Randomly select Nc classes for this client
        Nc = np.random.choice([2, 3])
        classes = np.random.choice(labels_list, Nc, replace=False)

        # Initialize weights and partition array
        L = len(labels_list)  # Total number of classes
        weights = np.zeros(L)  # Array to hold weights for each class
        P = np.zeros(L)  # Array to hold the number of samples for each class

        # Assign random weights to the selected classes
        for c in classes:
            weights[c] = np.random.random()  # Assign a random weight between 0 and 1
        total_weight = np.sum(weights)  # Sum of all weights
        num_samples = np.random.randint(Smin, Smax)

        # Sample data for the client
        client_indices = []  # List to store indices of data points for this client
        for c in classes:
            P[c] = int((weights[c] / total_weight) * num_samples)
            class_indices = np.where(labels.numpy() == c)[0]  # Indices of all samples belonging to class c
            num_class_samples = min(P[c], len(class_indices))  # Number of samples to select for this class
            sampled_indices = np.random.choice(class_indices, int(num_class_samples), replace=False)  # Randomly sample indices
            client_indices.extend(sampled_indices)

        # Store the client's data and labels
        dict_users[client_id] = client_indices

    return dict_users


