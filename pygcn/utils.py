import numpy as np
import scipy.sparse as sp
import torch


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(path="../data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    # First element is the index for the document
    # Final element is the label for the document
    # Elements in between are the one-hot word (type) features (i.e. bag of words)
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))

    # Extract the bag-of-words features as a "Compressed Sparse Row Matrix"
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # Extract the labels
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    # First column of idx_features_labels consists of the document IDs (i.e. the indices)
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)

    # Create a map/dict from the IDs to their position in the array
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)

    # Several things happening in this line:
    # * flatten the NUM_EDGES x 2 matrix;
    # * apply idx_map.get to replace the values in edges_unordered with their position in the array of indices (i.e. `idx`); and
    # * reshape the resulting matrix to be NUM_EDGES x 2 again
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)

    # According to the docs, a coo_matrix is:
    # "A sparse matrix in COOrdinate format. Also known as the 'ijv' or 'triplet' format."
    #
    # Here we are building it using this __init__:
    #
    #     coo_matrix((data, (i, j)), [shape=(M, N)])
    #         to construct from three arrays:
    #             1. data[:]   the entries of the matrix, in any order
    #             2. i[:]      the row indices of the matrix entries
    #             3. j[:]      the column indices of the matrix entries
    #
    #         Where ``A[i[k], j[k]] = data[k]``.  When shape is not
    #         specified, it is inferred from the index arrays
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # Row-normalize our feature and adjacency matrices
    features = normalize(features)
    # Note that sp.eye(N) gives us an identity matrix of dimensionality N
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
