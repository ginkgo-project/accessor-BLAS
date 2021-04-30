#!/usr/bin/env python3
import os
import csv
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_pdf import PdfPages


csv_files = ("./results/20210430_0600_v100_gemv_ab1_flops.csv",
             "./results/20210430_0600_v100_gemv_ab1_error.csv")
"""
csv_files = ("./results/20210430_0630_v100_gemv_a1b0_flops.csv",
             "./results/20210430_0630_v100_gemv_a1b0_error.csv")
"""

# TODO adjust FLOP computation to consider alpha and beta comps
def compute_flop(num_rows):
    return num_rows * (num_rows + 3) # with alpha and beta
    #return num_rows * num_rows # without alpha and beta


plot_folder = "./plots/"


### dictionary to match purpose to CSV header
h_dict_runtime = {
        "rows" : "Num Rows",
        "fp64": "GEMV fp64",
        "fp32": "GEMV fp32",
        "acc_mix": "GEMV Acc<fp64, fp32>",
        "cublas_fp64": "CUBLAS GEMV fp64",
        "cublas_fp32": "CUBLAS GEMV fp32",
        }

h_dict_error = {
        "rows" : "Num Rows",
        "fp64": "Error GEMV fp64",
        "fp32": "Error GEMV fp32",
        "acc_mix": "Error GEMV Acc<fp64, fp32>",
        "cublas_fp64": "Error CUBLAS GEMV fp64",
        "cublas_fp32": "Error CUBLAS GEMV fp32",
        }

def read_csv(h_dict, path=None):
    """
    Opens the CSV file in 'path' and returns 2 dictionaries:
    1. The key is the precision it was performed in, the value is the list of a list
       of column entries of the csv file (the lines are sorted according to number
       of computations)
    2. The key is the same as in h_dict, the value is the index of the row
       array / list for the correesponding key
    """
    if path == None:
        raise Exception("No filename specified! Unable to read file.")
    with open(path, 'r') as f:
        print("The csv file <{}> is opened".format(path))
        csv_f = csv.reader(f, delimiter=';', skipinitialspace=True)
        header = next(csv_f)
        print("CSV header: {}".format(header))
        
        i_dict = {}
        for key, val in h_dict.items():
            for i in range(len(header)):
                if header[i] == val:
                    i_dict[key] = i
        print("Resulting index dictionary: {}".format(i_dict))

        data = []

        for r in csv_f:
            data.append(r)

    return data, i_dict




############################### Actual Plotting ###############################
### Color definition
myblue    = (0, 0.4470, 0.7410); # acc
myorange  = (0.8500, 0.3250, 0.0980);
myyellow  = (0.9290, 0.6940, 0.1250);
mymagenta = (0.4940, 0.1840, 0.5560);
mygreen   = (0.4660, 0.6740, 0.1880); # sp
mycyan    = (0.3010, 0.7450, 0.9330);
myred     = (0.6350, 0.0780, 0.1840); # dp
myblack   = (0.2500, 0.2500, 0.2500);
mybrown   = (0.6500, 0.1600, 0.1600);

dark_mod = 2
mydarkred     = (0.6350 / dark_mod, 0.0780 / dark_mod, 0.1840 / dark_mod);
mydarkgreen   = (0.4660 / dark_mod, 0.6740 / dark_mod, 0.1880 / dark_mod);
mydarkblue    = (0, 0.4470 / dark_mod, 0.7410 / dark_mod);

### Other globals
LineWidth = 3
MarkerSize = 8

plot_order_flops = ["fp64", "fp32", "acc_mix", "cublas_fp64", "cublas_fp32"]
plot_order_error = ["fp32", "acc_mix", "cublas_fp64", "cublas_fp32"]

plot_dict = {
    "fp64": {
        "label": "GEMV fp64",
        "color": myred,
        "flops": [],
        "error": [],
        },
    "fp32": {
        "label": "GEMV fp32",
        "color": mygreen,
        "flops": [],
        "error": [],
        },
    "acc_mix": {
        "label": "GEMV Accessor<fp64, fp32>",
        "color": myblue,
        "flops": [],
        "error": [],
        },
    "cublas_fp64": {
        "label": "GEMV CuBLAS fp64",
        "color": myorange,
        "flops": [],
        "error": [],
        },
    "cublas_fp32": {
        "label": "GEMV CuBLAS fp32",
        "color": mymagenta,
        "flops": [],
        "error": [],
        },
    }



def create_fig_ax():
    """
    Creates a tuple of figure and axis for future plots.
    The size, the visibility of the grid and the log-scale of x and y is preset
    """
    fig = Figure(figsize=(10, 4)) # Properly garbage collected
    ax = fig.add_subplot()
    #fig, ax = plt.subplots(figsize=(10, 4)) # NOT garbage collected!
    grid_minor_color = (.9, .9, .9)
    grid_major_color = (.8, .8, .8)
    ax.grid(True, which="major", axis="both", linestyle='-', linewidth=1, color=grid_major_color)
    ax.grid(True, which="minor", axis="both", linestyle=':', linewidth=1, color=grid_minor_color)
    #ax.loglog()
    return fig, ax


def plot_figure(fig, file_name, plot_prefix):
    """Plots the given figure fig as various formats with a base-name of file_name.
    plot_folder will be used as the filder for the file; plot_prefix will be the
    prefix of each file."""

    file_path = plot_folder + plot_prefix + file_name
    p_bbox = "tight"
    p_pad = 0
    p_dpi = 300  # Only useful for non-scalable formats
    with PdfPages(file_path+".pdf") as export_pdf:
        export_pdf.savefig(fig, dpi=p_dpi, bbox_inches=p_bbox, pad_inches=p_pad)
    fig.savefig(file_path+".svg", dpi=p_dpi, bbox_inches=p_bbox, pad_inches=p_pad, format="svg")
    #fig.savefig(file_path+".png", dpi=p_dpi, bbox_inches=p_bbox, pad_inches=p_pad, format="png")


if __name__ == "__main__":
    # Change to the directory where the script is placed
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Make sure the plot folder exists
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    ### Plot FLOPs
    data, i_dict = read_csv(h_dict_runtime, csv_files[0])

    num_rows = []
    
    for line in data:
        if (len(line) < 2):
            break;
        rows = int(line[i_dict["rows"]])
        num_rows.append(rows)
        mflops = compute_flop(rows) / (1000 * 1000)
        
        for name, info_dict in plot_dict.items():
            time_ms = float(line[i_dict[name]])
            info_dict["flops"].append(mflops / time_ms)

    fig, ax = create_fig_ax()

    ax.set_xlabel("Number of rows")
    ax.set_ylabel("GFLOP/s")
    for name in plot_order_flops:
        info = plot_dict[name]
        ax.plot(num_rows, info["flops"], label=info["label"],
                marker='', color=info["color"], linewidth=LineWidth,
                markersize=MarkerSize)
    #ax.legend(loc="best")
    ax.legend(loc="lower right")
    plot_figure(fig, "gemv_flops", "v100_")
    
    ### Plot Error
    data, i_dict = read_csv(h_dict_error, csv_files[1])

    num_rows = []
    
    for line in data:
        if (len(line) < 2):
            break;
        rows = int(line[i_dict["rows"]])
        num_rows.append(rows)
        
        for name, info_dict in plot_dict.items():
            error = float(line[i_dict[name]])
            info_dict["error"].append(error)

    fig, ax = create_fig_ax()
    ax.set_yscale('log')

    ax.set_xlabel("Number of rows")
    ax.set_ylabel("Relative error")
    for name in plot_order_error:
        info = plot_dict[name]
        ax.plot(num_rows, info["error"], label=info["label"],
                marker='', color=info["color"], linewidth=LineWidth,
                markersize=MarkerSize)
    #ax.legend(loc="best")
    ax.legend(loc="lower right")
    plot_figure(fig, "gemv_error", "v100_")



