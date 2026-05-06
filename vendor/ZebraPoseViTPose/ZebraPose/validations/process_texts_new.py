import os
import re
import sys

import ipdb
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

# Define the folder path where your text files are located
folder_path = sys.argv[1]

# Initialize an empty dictionary to store results
result_dict = {}

patterns = ['_300', '_zoo', '_ap10', '_ap10_NZ', '_ap10_OZ', '_apt36k', '_apt36k_NZ', '_apt36k_OZ', '_grevy_zebras', '_unsup_dom_ad_horse','_learn_syn_an_horse']
re_patterns = [re.compile(pattern+".txt") for pattern in patterns]

pattern_either = re.compile(r'(PCK@0.05|PCK@0.1|PCK@0.2|AP:|AP .5:)')

# Iterate over all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
        file_path = os.path.join(folder_path, filename)

        found = False
        for p in re_patterns:
            if p.search(filename):
                category = p.search(filename).group()
                found = True
                break
        if not found:
            import ipdb; ipdb.set_trace()

        # Initialize a dictionary to store lines by category
        if category not in result_dict:
            result_dict[category] = {}

        # Initialize a list to store lines containing the desired pattern
        matching_lines = []

        # Open and read the file line by line
        with open(file_path, 'r') as file:
            lines = file.read().splitlines()
            line_id = 0
            while line_id < len(lines):
                line = lines[line_id]
                if "size mismatch" in line or "error" in line:# or ("27" in line and "17" in line):
                    print(f"error {file_path}")
                    ipdb.set_trace()

                # Check if the line contains the pattern for "either"
                if pattern_either.search(line):
                    if "pck" in line.lower():
                        thr = line.split("@")[1]
                        line_id += 1
                        line = lines[line_id]
                        line = line.replace("[(","").replace(")]","").replace(":","").replace("'","").replace(",","").replace("AP .5", "AP50")
                        line = line.replace("PCK", f"PCK@{thr}")
                        line = line.replace("(","")
                        line = line.replace(")","")
                        matching_lines.append(line.strip())
                        line_id += 1
                        tmp_line = ""
                        while "]))" not in line:
                            line = lines[line_id]
                            tmp_line += line.replace("[(","").replace(")]","").replace(":","").replace("'","").replace(",","").replace("AP .5", "AP50")
                            line_id += 1
                        while "  " in tmp_line:
                            tmp_line = tmp_line.replace("  ", " ")
                        tmp_line = tmp_line.replace("(","").replace(")","")
                        tmp_line = tmp_line.replace("PCK_parts", f"PCK_parts@{thr}")
                        matching_lines.append(tmp_line.strip())
                    else:
                        line = line.replace("[(","").replace(")]","").replace(":","").replace("'","").replace(",","").replace("AP .5", "AP50")
                        matching_lines.append(line.strip())
                        line_id += 1
                else:
                    line_id += 1
        # Add the matching lines to the result dictionary with the filename as the key
        result_dict[category][filename] = matching_lines

import ipdb;

#ipdb.set_trace()

# Print the result dictionary
for category, files in result_dict.items():
    print(f"Category: {category}")
    for filename, lines in files.items():
        if "1s" in filename or "1250" in filename or "synap" in filename or filename.startswith("UDA_99") or filename.startswith("pre_UDA_99") or filename.endswith("NZ.txt") or "NO_Z" in filename or "3k" in filename:
            continue
        print(f"File: {filename}")
        for line in lines:
            if "parts" not in line.lower() and ("PCK@0.05" in line or "PCK@0.1" in line):
                print(line)
            elif "PCK_parts@0.05" in line:
                if ("_s_" in filename or "s_zebras" in filename or "s_1s" in filename or "mix" in filename or "s_" in filename):
                    line.replace("PCK_parts@0.05", "PCK@0.05_red")
                    line.replace("array", "")
                    tmp = line.split('[')[1][:-1].strip().split(" ")
                    s = 0
                    cnt = 0
                    for id_val, val in enumerate(tmp):
                        if "-1" not in val and id_val not in [2,5,8,11,13] and val != '':
                            s += float(val)
                            cnt += 1
                    print("PCK@0.05_red ", s/cnt)
                else:
                    print("PCK@0.05_red ", 0.000)
            elif "PCK_parts@0.1" in line:
                if ("_s_" in filename or "s_zebras" in filename or "s_1s" in filename or "mix" in filename or "s_" in filename):
                    line.replace("PCK_parts@0.1", "PCK@0.1_red")
                    line.replace("array", "")
                    tmp = line.split('[')[1][:-1].strip().split(" ")
                    s = 0
                    cnt = 0
                    for id_val, val in enumerate(tmp):
                        if "-1" not in val and id_val not in [2,5,8,11,13] and val != '':
                            s += float(val)
                            cnt += 1
                    print("PCK@0.1_red ", s/cnt)
                else:
                    print("PCK@0.1_red ", 0.000)
            elif "AP50 " in line:
                line = line.replace("AP50","AP")
                print(line)
        print()

np.save('results.npy', result_dict)

if len(sys.argv) > 2:
    sel_train_set = sys.argv[2:]
else:
    sel_train_set = [""]

for category, files in result_dict.items():
    # set up a plot with 2 rows, 3 columns in the first row, 2 columns in the second one

    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(category)

    for id_plot, metric in enumerate(['PCK@0.05','PCK@0.1', 'PCK@0.2', 'AP', 'AP50']):
        ax = axs[id_plot//3, id_plot%3]
        ax.set_title(metric)

        x = []  # Initialize an empty list for x-values
        y = []  # Initialize an empty list for y-values
        labels = []  # Initialize an empty list for labels
        counter = 0  # Initialize a counter

        for filename, lines in files.items():
            found = False
            for p in sel_train_set:
                if (("mix" in filename and (p in filename.split("_")[1] or p in filename.split("_")[2] or p in filename.split("_")[3])) or \
                    ("mix" not in filename and p in "".join(filename.split("_")[:2]))) or\
                    ("mix" not in filename and ("_s_" in filename or "s_zebras" in filename or "s_1s" in filename or "s_" in filename)):
                    found = True
                    break
            if not found and len(sel_train_set) > 0:
                continue
            for line in lines:
                if ("_s_" in filename or "s_zebras" in filename or "s_1s" in filename or "mix" in filename or "s_" in filename) and "parts" in line.split()[0] and metric in line.replace("_parts",""):
                    x.append(counter)
                    vals = line.split()[2:][:-1]
                    s = 0
                    cnt = 0
                    for id_val, val in enumerate(vals):
                        if "-1" not in val and id_val not in [2,5,8,11,13]:
                            s += float(val)
                            cnt += 1
                    y.append(s/cnt)
                    labels.append(filename.split("_old_adam")[0].replace("pre_","")+"_reduced")
                    counter += 1

                if line.split()[0] == metric:
                    x.append(counter)  # Use the counter as x-value
                    y.append(float(line.split()[1]))
                    labels.append(filename.split("_old_adam")[0].replace("pre_", "").split("_oa")[0])
                    counter += 1  # Increment the counter

        ax.scatter(x, y)
        for i, txt in enumerate(labels):
            ax.annotate(txt, (x[i], y[i]))
        
        # Add labels for each point
        for i, txt in enumerate(labels):
            plt.annotate(txt, (x[i], y[i]))
        ax.set(xlabel="Experiment", ylabel=metric)
        # ax.legend()
        ax.grid(True)
    fig.delaxes(axs[1,2])
    plt.show()





