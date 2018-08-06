from pathlib import Path
import sys
import subprocess


def record_settings(out, args, dataset_conf):
    """
    record command line arguments and dataset information
    """
    out = Path(out)
    if out.exists() is False:
        out.mkdir(parents=True, exist_ok=True)
    # subprocess.call("cp *.py %s" % out, shell=True)

    config_file = out / 'configurations.txt'
    with open(config_file, "w") as f:

        # write arguments
        f.write('Arguments' + '\n')
        for key, value in vars(args).items():
            f.write(key + '\t' + str(value) + '\n')

        # write Dataset configurations
        f.write('\n\n')
        f.write('Dataset' + '\n')
        for key, value in dataset_conf.items():
            f.write(key + '\t' + str(value) + '\n')
