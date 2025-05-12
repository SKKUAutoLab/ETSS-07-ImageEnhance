from .multi_read_data import *


def CreateDataset(args, task):
    dataset = None
    if "dataset" not in args:
        dataset = DefaultDataset()
    elif "RLV" == args.dataset:
        dataset = RLVDataLoader()
    elif "DID" == args.dataset:
        dataset = DidDataloader()
    else:
        dataset = DefaultDataset()
    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(args, task)
    return dataset
