import json
import os
from collections import OrderedDict
from pathlib import Path
from bunch import Bunch
from random import randint
from utils.dirs import create_dirs

class Bunch_plus(Bunch):
    def __init__(self, *args, **kwargs):
        super(Bunch_plus, self).__init__(*args, **kwargs)

    def print(self):
        line = "-" * 92
        print(line + "\n" +
              "| {:^35s} | {:^50} |\n".format('Feature', 'Value') +
              "=" * 92)
        for key, val in self.items():
            if isinstance(val, OrderedDict):
                raise NotImplementedError("Nested configs are not implemented")
            else:
                print("| {:35s} | {:50} |\n".format(key, str(val)) + line)
        print("\n")

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        config_dict = json.load(handle, object_hook=OrderedDict)
        config = Bunch_plus(config_dict)
        return config


def process_config(args):
    if args.config is not None:
        json_file = args.config
    else:
        raise ValueError("preprocess config: config path wasn't specified")

    config = read_json(json_file)
    for arg in sorted(vars(args)):
        key = arg
        val = getattr(args, arg)
        if val is not None:
            setattr(config, key, val)

    if args.seed is None and config.seed is None:
        config.seed = randint(0, 1000000)

    config.experiment_dir = os.path.join("./experiments",
                                         config.exp_name)
    config.summary_dir = os.path.join(config.experiment_dir, "summary")
    config.checkpoint_dir = os.path.join(config.experiment_dir, "checkpoint")
    config.print()


    # create the experiments dirs
    create_dirs(config.experiment_dir)

    return config
