import os
import sys
import shutil

def create_dirs(experiment_dir):
    """
    dirs - a list of directories to create if these directories are not found
    :param dirs:
    :return exit_code: 0:success -1:failed
    """
    try:
        summary_dir = os.path.join(experiment_dir, "summary")
        checkpoint_dir = os.path.join(experiment_dir, "checkpoint")
        scripts_dir = os.path.join(experiment_dir, "scripts")

        scripts_to_save = list()
        for root, subdirs, files in os.walk('.'):
            for file in files:
                filename = os.fsdecode(file)
                if filename.endswith(".py") or filename.endswith(".json"):
                    scripts_to_save.append(os.path.join(root, filename))


        if not os.path.exists(summary_dir):
            os.makedirs(summary_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        if scripts_to_save is not None:
            if not os.path.exists(scripts_dir):
                os.makedirs(scripts_dir)

                for script in scripts_to_save:
                    dst_file = os.path.join(scripts_dir, os.path.basename(script))
                    shutil.copyfile(os.path.join(os.path.dirname(sys.argv[0]), script), dst_file)

    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)
