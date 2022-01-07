import argparse
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from tqdm import tqdm
import feather

parser: ArgumentParser = argparse.ArgumentParser()
parser.add_argument("--data_path",
                    help='Specify path to the case samples data in csv format (patients as rows, genes as columns)',
                    default="tests/test_case.csv",
                    required=False)
parser.add_argument("--network_path", help="Specify path to the network file in csv format with two columns",
                    default="tests/test_network.csv",
                    required=False)
parser.add_argument("--out_path", help="Specify path to the algorithm results",
                    default="tests/lioness_output",
                    required=False)
parser.add_argument("--threshold", help="Define a quantile threshold on edges to make them binary", default=0.99,
                    required=False)
data_path = "tests/test_case.csv"

network_path = "tests/test_network.csv"
out_path = "tests/lioness"
args = parser.parse_args()


def run_lioness(data_path, network_path, out_path, threshold=0.99):
    case = pd.read_csv(data_path, index_col=0)
    patients = case.index
    normal_net = pd.read_csv(network_path)
    genes = list(case.columns)
    genes_dict = {genes[i]: i for i in range(len(genes))}
    case = case.T
    cor = np.corrcoef(case)
    crc = np.zeros((len(patients), normal_net.shape[0]))
    pat_count = 0
    for pat in tqdm(patients):
        data_p = case.copy().drop(columns=[pat])
        corr_p = np.corrcoef(data_p)
        net = len(patients) * (cor - corr_p) + corr_p
        q = np.quantile(net[np.triu_indices(net.shape[0], k=1)], threshold)
        net = net > q
        edge_count = 0
        edges = []
        for tup in normal_net.itertuples():
            if tup[1] in genes_dict and tup[2] in genes_dict:
                edge = (genes_dict[tup[1]], genes_dict[tup[2]])
                if net[edge]:
                    crc[pat_count, edge_count] = 1

            edge_count += 1
            edges.append((tup[1], tup[2]))

        pat_count += 1
    edges = [str(x) for x in edges]
    data = pd.DataFrame(crc, columns=edges)
    data['patient id'] = patients

    data.to_feather(out_path)


run_lioness(args.data_path, args.network_path, args.out_path, args.threshold)
