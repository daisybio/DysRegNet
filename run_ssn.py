import argparse
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import feather
from scipy import stats
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

parser: ArgumentParser = argparse.ArgumentParser()
parser.add_argument("--case_path",
                    help='Specify path to the case samples data in csv format (patients as rows, genes as columns)',
                    default="tests/test_case.csv",
                    required=False)
parser.add_argument("--control_path",
                    help='Specify path to the control samples data in csv format (patients as rows, genes as columns)',
                    default="tests/test_control.csv",
                    required=False)
parser.add_argument("--network_path", help="Specify path to the network file in csv format with two columns",
                    default="tests/test_network.csv",
                    required=False)
parser.add_argument("--out_path", help="Specify path to the algorithm results",
                    default="tests/ssn_output",
                    required=False)
parser.add_argument("--pvalue", help="Define a p-value threshold for edges", default=0.005,
                    required=False)

args = parser.parse_args()


def run_ssn(case_path, control_path, network_path, out_path, pvalue=0.005):
    case = pd.read_csv(case_path, index_col=0)
    control = pd.read_csv(control_path, index_col=0)
    normal_net = pd.read_csv(network_path)
    genes = list(case.columns)
    genes_dict = {genes[i]: i for i in range(len(genes))}
    patients = list(case.index)
    n = len(patients)
    genes = set(genes)
    control = control.T
    cor = np.corrcoef(control)
    crc = np.zeros((len(patients), normal_net.shape[0]))
    pat_count = 0
    for pat in tqdm(patients):
        data_p = control.copy()
        data_p[pat] = case.loc[pat]
        corr_p = np.corrcoef(data_p)
        dcor = corr_p - cor
        zscore = dcor / ((1 - cor ** 2) / (n - 1))
        pv = stats.norm.sf(zscore)
        pv = pv < pvalue / normal_net.shape[0]
        edge_count = 0
        edges = []
        for tup in normal_net.itertuples():
            if tup[1] in genes and tup[2] in genes:

                edge = (genes_dict[tup[1]], genes_dict[tup[2]])
                if pv[edge]:
                    crc[pat_count, edge_count] = 1

            edge_count += 1
            edges.append((tup[1], tup[2]))

        pat_count += 1
    edges = [str(x) for x in edges]
    print(len(edges))
    data = pd.DataFrame(crc, columns=edges)
    data['patient id'] = patients
    data.to_feather(out_path)


run_ssn(args.case_path, args.control_path, args.network_path, args.out_path, args.pvalue)
