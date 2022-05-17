import argparse


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--batch_size',          default=None, type=int, help='batch size')
    argparser.add_argument('--batch_size_eval',     default=None, type=int, help='batch size eval')
    argparser.add_argument('--channel_cardinality', default=None, type=int, help='channel size')
    argparser.add_argument('--config',              default=None, type=str, help='configuration file')
    argparser.add_argument('--channel',             default=None, type=str, help='channel name')
    argparser.add_argument('--exp_name',            default=None, type=str, help='experiment name')
    argparser.add_argument('--run_name',            default=None, type=str, help='run name')
    argparser.add_argument('--n_clusters',          default=None, type=int, help='clusters for state space')
    argparser.add_argument('--seed',                default=None, type=int, help='randomization seed')
    argparser.add_argument('--unroll_steps',        default=None, type=int, help='time unroll steps')

    args = argparser.parse_args()
    return args
