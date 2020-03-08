import argparse


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--channel_cardinality', default=None, type=int, help='channel_size')
    argparser.add_argument('--config',              default=None, type=str, help='configuration file')
    argparser.add_argument('--exp_name',            default=None, type=str, help='experiment name')
    argparser.add_argument('--n_clusters',          default=None, type=int, help='clusters for state space')
    argparser.add_argument('--seed',                default=None, type=int, help='randomization seed')

    args = argparser.parse_args()
    return args
