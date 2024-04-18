"""
Command-line interface for running privacy evaluation with respect to the risk of linkability
"""

import json

from os import mkdir, path
from numpy.random import choice, seed
from argparse import ArgumentParser
from pandas import DataFrame

from utils.datagen import load_s3_data_as_df, load_local_data_as_df
from utils.utils import json_numpy_serialzer
from utils.logging import LOGGER
from utils.constants import *

from feature_sets.model_agnostic import NaiveFeatureSet, EnsembleFeatureSet
from feature_sets.bayes import CorrelationsFeatureSet

from generative_models.data_synthesiser import (BayesianNet, PrivBayes)

from warnings import simplefilter
simplefilter('ignore', category=FutureWarning)
simplefilter('ignore', category=DeprecationWarning)

cwd = path.dirname(__file__)


SEED = 42


def main():
    argparser = ArgumentParser()
    datasource = argparser.add_mutually_exclusive_group()
    datasource.add_argument('--s3name', '-S3', type=str, choices=['adult', 'census', 'credit', 'alarm', 'insurance'], help='Name of the dataset to run on')
    datasource.add_argument('--datapath', '-D', default='data\\texas', type=str, help='Relative path to cwd of a local data file')
    argparser.add_argument('--runconfig', '-RC', default='tests\\linkage\\runconfig.json', type=str, help='Path relative to cwd of runconfig file')
    argparser.add_argument('--outdir', '-O', default='tests\\linkage', type=str, help='Path relative to cwd for storing output files')
    args = argparser.parse_args()

    # Load runconfig
    with open(path.join(cwd, args.runconfig)) as f:
        runconfig = json.load(f)
    print('Runconfig:')
    print(runconfig)

    # Load data
    if args.s3name is not None:
        rawPop, metadata = load_s3_data_as_df(args.s3name)
        dname = args.s3name
    else:
        rawPop, metadata = load_local_data_as_df(path.join(cwd, args.datapath))
        dname = args.datapath.split('/')[-1]

    print(f'Loaded data {dname}:')
    print(rawPop.info())

    # Make sure outdir exists
    if not path.isdir(args.outdir):
        mkdir(args.outdir)

    seed(SEED)

    ########################
    #### GAME INPUTS #######
    ########################
    # Pick targets
    targetIDs = choice(list(rawPop.index), size=runconfig['nTargets'], replace=False).tolist()

    # If specified: Add specific target records
    if runconfig['Targets'] is not None:
        targetIDs.extend(runconfig['Targets'])

    targets = rawPop.loc[targetIDs, :]

    # Drop targets from population
    rawPopDropTargets = rawPop.drop(targetIDs)

    # Init adversary's prior knowledge
    rawAidx = choice(list(rawPopDropTargets.index), size=runconfig['sizeRawA'], replace=False).tolist()
    rawA = rawPop.loc[rawAidx, :]

    # List of candidate generative models to evaluate
    gmList = []
    if 'generativeModels' in runconfig.keys():
        for gm, paramsList in runconfig['generativeModels'].items():
            if gm == 'BayesianNet':
                for params in paramsList:
                    gmList.append(BayesianNet(metadata, *params))
            elif gm == 'PrivBayes':
                for params in paramsList:
                    gmList.append(PrivBayes(metadata, *params))
            else:
                raise ValueError(f'Unknown GM {gm}')
            
if __name__ == "__main__":
    main()