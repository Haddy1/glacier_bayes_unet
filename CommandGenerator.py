from pathlib import Path

program = 'python3 main.py'
nr_parallel_cmds = 4    # nr of scripts generated
out_path = 'output'     # where results should be written to
identifier = 'combined_loss'
patch_size = 256
arguments = {
    'loss': 'combined_loss',
    'loss_parms' : {
        'binary_crossentropy':[1.0,0.8,0.6,0.4,0.2,0.0],
        'focal_loss': [0.0,0.2,0.4,0.6,0.8,1.0]
    },
    'batch_size':16,
    'data_path': 'data_' + str(patch_size),
}

# Don't change anything below this line

def addArgument(cmds, arg, value):
    new_cmds = []
    for cmd in cmds:
        cmd = cmd + " --" + str(arg)

        # List
        if isinstance(value, list):
            for item in value:
                new_cmds.append(cmd + " " + str(item))

        # Dictionary
        elif isinstance(value, dict):
            list_items = False           # flag for checking if list items are in dicionary
            no_list_items = False   # flag to for checking if non list items are in dictionary
            dict_cmds = []
            first = True        # for list items indicates whether first entry was processed
            for key, item in value.items():
                if isinstance(item, list):
                    list_items = True
                    if first:
                        first = False
                        for entry in item:
                            dict_cmds.append(cmd + " " + str(key) + "=" + str(entry))
                    else: # extend new cmd
                        for i, entry in enumerate(item):
                            dict_cmds[i] += "," + str(key) + "=" + str(entry)
                else:
                    no_list_items = True
                    cmd += " " + str(key) + "=" + str(item)

            if list_items and no_list_items:
                raise IndexError("parameter lists in dictionary items are not of the same size")
            elif list_items:
                new_cmds += dict_cmds
            else:
                new_cmds.append(cmd)


        # Single value
        else:
            new_cmds.append(cmd + " " + str(value))

    return new_cmds


cmds = [program]
for arg, value in arguments.items():
    cmds = addArgument(cmds, arg, value)


# Write bash script file for each job
cmd_per_job = len(cmds) // nr_parallel_cmds
for j_id in range(nr_parallel_cmds):
    with open('run_' + identifier + '_' + str(j_id) + '.sh', 'w') as script:
        # header
        script.write(
            "#!/bin/bash\n" +
            "#SBATCH --job-name=Glacier" + str(j_id) + "\n" +
            "#SBATCH --ntasks=1\n" +
            "#SBATCH --cpus-per-task=2\n" +
            "#SBATCH --mem=12000\n" +
            "#SBATCH --gres=gpu:1\n" +
            "#SBATCH -o " + identifier + "_" + str(j_id) + ".out\n" +
            "#SBATCH -e " + identifier + "_" + str(j_id) + ".err\n"
        )

        for cmd in cmds[j_id * cmd_per_job:cmd_per_job]:
            out = Path(out_path, identifier)
            script.write("\noutput=" + str(out) + "_`date +%m%d-%H%M%S`\n")     # output dir
            script.write(cmd + " --$out\n")



