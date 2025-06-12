import pprint
from encode.utils import read_log, enc_selector
import argparse


#Example of how to run:
#python3 main.py --dataset event_logs/helpdesk.xes --encoding node2vec --vector_size 128

def read_args():
    args = argparse.ArgumentParser()
    args.add_argument("--dataset", type=str, default="scenario1_1000_attribute_0.05.xes")
    args.add_argument("--encoding", type=str, default="onehot")
    args.add_argument("--vector_size", type=int, default=8)
    args.add_argument("--aggregation", type=str, default="average")
    args.add_argument("--embed_from", type=str, default="nodes")
    args.add_argument("--edge_operator", type=str, default="average")

    return args.parse_args()


def run(config):
    log = read_log(config["dataset"])

    encoder = enc_selector(config["encoding"])
    encoding = encoder(config, log)
    encoding.to_csv(f'{config["encoding"]}_{config["vector_size"]}d-{config["dataset"].split("/")[-1].replace(".xes","")}--dfg.csv', index=False)


if __name__ == "__main__":
    config = read_args()
    config = vars(config)

    print("\n\nConfig:")
    pprint.pprint(config)

    print("Running...")
    run(config)
