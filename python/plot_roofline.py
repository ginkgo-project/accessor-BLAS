#!/usr/bin/env python3
import os
import csv
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_pdf import PdfPages


# TODO: Add marker at different points to better distinguish the lines

plot_folder = "./plots/"


### dictionary to match purpose to CSV header
h_dict_gemv_runtime = {
        "size" : "Num Rows",
        "fp64": "GEMV fp64",
        "fp32": "GEMV fp32",
        "acc_mix": "GEMV Acc<fp64, fp32>",
        "cublas_fp64": "CUBLAS GEMV fp64",
        "cublas_fp32": "CUBLAS GEMV fp32",
        }
h_dict_gemv_runtime2 = h_dict_gemv_runtime.copy()
h_dict_gemv_runtime2["size"] = "Num rows"

h_dict_gemv_error = {
        "size" : "Num Rows",
        "fp64": "Error GEMV fp64",
        "fp32": "Error GEMV fp32",
        "acc_mix": "Error GEMV Acc<fp64, fp32>",
        "cublas_fp64": "Error CUBLAS GEMV fp64",
        "cublas_fp32": "Error CUBLAS GEMV fp32",
        }
h_dict_gemv_error2= h_dict_gemv_error.copy()
h_dict_gemv_error2["size"] = "Num rows"

h_dict_dot_runtime = {
        "size" : "Vector Size",
        "fp64": "DOT fp64",
        "fp32": "DOT fp32",
        "acc_mix": "DOT Acc<fp64, fp32>",
        "cublas_fp64": "CUBLAS DOT fp64",
        "cublas_fp32": "CUBLAS DOT fp32",
        }

h_dict_dot_error = {
        "size" : "Vector Size",
        "fp64": "Error DOT fp64",
        "fp32": "Error DOT fp32",
        "acc_mix": "Error DOT Acc<fp64, fp32>",
        "cublas_fp64": "Error CUBLAS DOT fp64",
        "cublas_fp32": "Error CUBLAS DOT fp32",
        }

h_dict_trsv_runtime = {
        "size" : "Num rows",
        "fp64": "TRSV fp64",
        "fp32": "TRSV fp32",
        "acc_mix": "TRSV Acc<fp64, fp32>",
        "cublas_fp64": "CUBLAS TRSV fp64",
        "cublas_fp32": "CUBLAS TRSV fp32",
        }

h_dict_trsv_error = {
        "size" : "Num rows",
        "fp64": "Error TRSV fp64",
        "fp32": "Error TRSV fp32",
        "acc_mix": "Error TRSV Acc<fp64, fp32>",
        "cublas_fp64": "Error CUBLAS TRSV fp64",
        "cublas_fp32": "Error CUBLAS TRSV fp32",
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
LabelFontSize = 15
AxisTickSize = 12

PlotLineWidth = 3
MarkerSize = 10
TotalMarkerPerPlot = 5

plot_order_flops = ["fp64", "fp32", "acc_mix", "cublas_fp64", "cublas_fp32"]
plot_order_error = ["fp32", "acc_mix", "cublas_fp64", "cublas_fp32"]

plot_detail_dict = {
    "fp64": {
        "label": "fp64",
        "color": myred,
        "marker": 'X',
        "zorder": 3.1,
        },
    "fp32": {
        "label": "fp32",
        "color": mygreen,
        "marker": 'P',
        "zorder": 3.2,
        },
    "acc_mix": {
        "label": "Accessor<fp64, fp32>",
        "color": myblue,
        "marker": 'x',
        "zorder": 3.3,
        },
    "cublas_fp64": {
        "label": "cuBLAS fp64",
        "color": myorange,
        "marker": 'd',
        "zorder": 3.4,
        },
    "cublas_fp32": {
        "label": "cuBLAS fp32",
        "color": mymagenta,
        "marker": '+',
        "zorder": 3.5,
        },
    }


def gemv_compute_flop(size, time_ms):
    flops = size * (size + 3) # with alpha and beta
    #flops = size * size # without alpha and beta
    Mflops = flops / (1000 * 1000)
    return Mflops / time_ms

def gemv_compute_error(size, error):
    return error

def trsv_compute_flop(size, time_ms):
    flops = size * size
    Mflops = flops / (1000 * 1000)
    return Mflops / time_ms

def dot_compute_flop(size, time_ms):
    return ((2*size - 1) / (1000 * 1000)) / time_ms


plot_dict_list = [
        {
            #"file": "./results/20210430_0600_v100_gemv_ab1_flops.csv",
            #"file": "./results/20210430_2000_v100_gemv_ab1_flops.csv",
            #"file": "./results/20210505_1300_v100_gemv_ab1_flops.csv",
            #"file": "./results/20210506_1600_v100_gemv_ab1_flops_-1,1.csv",
            #"file": "./results/20210506_1800_v100_gemv_ab1_flops_0,1.csv",
            "file": "./results/20210524_1739_v100_gemv_time_ms.csv",
            "header_trans": h_dict_gemv_runtime,
            "plot_order": plot_order_flops,
            "plot_detail": plot_detail_dict,
            "plot_name": "gemv_flops",
            "plot_prefix": "v100_",
            "label_prefix": "GEMV ",
            "conv_func": gemv_compute_flop,
            "xlabel": "Number of rows",
            "ylabel": "GFLOP/s",
            "yscale": "linear",
            #"xlim": {"left": None, "right": None,},
            #"ylim": {"bottom": 0, "top": 225,},
        },
        {
            #"file": "./results/20210430_0600_v100_gemv_ab1_error.csv",
            #"file": "./results/20210430_2000_v100_gemv_ab1_error.csv",
            #"file": "./results/20210505_1300_v100_gemv_ab1_error.csv",
            #"file": "./results/20210506_1600_v100_gemv_ab1_error_-1,1.csv",
            #"file": "./results/20210506_1800_v100_gemv_ab1_error_0,1.csv",
            "file": "./results/20210524_1739_v100_gemv_error.csv",
            "header_trans": h_dict_gemv_error,
            "plot_order": plot_order_error,
            "plot_detail": plot_detail_dict,
            "plot_name": "gemv_error",
            "plot_prefix": "v100_",
            "label_prefix": "GEMV ",
            "conv_func": gemv_compute_error,
            "xlabel": "Number of rows",
            "ylabel": "Relative error",
            "yscale": "log",
        },
        {
            "file": "./results/20210829_1839_v100_gemv_time_ms_0,1.csv",
            "header_trans": h_dict_gemv_runtime2,
            "plot_order": plot_order_flops,
            "plot_detail": plot_detail_dict,
            "plot_name": "gemv_flops_0,1",
            "plot_prefix": "v100_",
            "label_prefix": "GEMV ",
            "conv_func": gemv_compute_flop,
            "xlabel": "Number of rows",
            "ylabel": "GFLOP/s",
            "yscale": "linear",
            #"xlim": {"left": None, "right": None,},
            #"ylim": {"bottom": 0, "top": 225,},
        },
        {
            "file": "./results/20210829_1839_v100_gemv_error_0,1.csv",
            "header_trans": h_dict_gemv_error2,
            "plot_order": plot_order_error,
            "plot_detail": plot_detail_dict,
            "plot_name": "gemv_error_0,1",
            "plot_prefix": "v100_",
            "label_prefix": "GEMV ",
            "conv_func": gemv_compute_error,
            "xlabel": "Number of rows",
            "ylabel": "Relative error",
            "yscale": "log",
        },
        {
            "file": "./results/20210526_1201_a100_gemv_time_ms.csv",
            "header_trans": h_dict_gemv_runtime,
            "plot_order": plot_order_flops,
            "plot_detail": plot_detail_dict,
            "plot_name": "gemv_flops",
            "plot_prefix": "a100_",
            "label_prefix": "GEMV ",
            "conv_func": gemv_compute_flop,
            "xlabel": "Number of rows",
            "ylabel": "GFLOP/s",
            "yscale": "linear",
        },
        {
            "file": "./results/20210526_1201_a100_gemv_error.csv",
            "header_trans": h_dict_gemv_error,
            "plot_order": plot_order_error,
            "plot_detail": plot_detail_dict,
            "plot_name": "gemv_error",
            "plot_prefix": "a100_",
            "label_prefix": "GEMV ",
            "conv_func": gemv_compute_error,
            "xlabel": "Number of rows",
            "ylabel": "Relative error",
            "yscale": "log",
        },
        {
            #"file": "./results/20210524_1739_v100_trsv_time_ms.csv",
            "file": "./results/20210525_1746_v100_trsv_time_ms.csv",
            "header_trans": h_dict_trsv_runtime,
            "plot_order": plot_order_flops,
            "plot_detail": plot_detail_dict,
            "plot_name": "trsv_flops",
            "plot_prefix": "v100_",
            "label_prefix": "TRSV ",
            "conv_func": trsv_compute_flop,
            "xlabel": "Number of rows",
            "ylabel": "GFLOP/s",
            "yscale": "linear",
            #"xlim": {"left": None, "right": None,},
            #"ylim": {"bottom": 0, "top": 225,},
        },
        {
            #"file": "./results/20210524_1739_v100_trsv_error.csv",
            "file": "./results/20210525_1746_v100_trsv_error.csv",
            "header_trans": h_dict_trsv_error,
            "plot_order": plot_order_error,
            "plot_detail": plot_detail_dict,
            "plot_name": "trsv_error",
            "plot_prefix": "v100_",
            "label_prefix": "TRSV ",
            "conv_func": lambda sz, error: error,
            "xlabel": "Number of rows",
            "ylabel": "Relative error",
            "yscale": "log",
        },
        {
            "file": "./results/20210829_1839_v100_trsv_time_ms_0,1.csv",
            "header_trans": h_dict_trsv_runtime,
            "plot_order": plot_order_flops,
            "plot_detail": plot_detail_dict,
            "plot_name": "trsv_flops_0,1",
            "plot_prefix": "v100_",
            "label_prefix": "TRSV ",
            "conv_func": trsv_compute_flop,
            "xlabel": "Number of rows",
            "ylabel": "GFLOP/s",
            "yscale": "linear",
            #"xlim": {"left": None, "right": None,},
            #"ylim": {"bottom": 0, "top": 225,},
        },
        {
            "file": "./results/20210829_1839_v100_trsv_error_0,1.csv",
            "header_trans": h_dict_trsv_error,
            "plot_order": plot_order_error,
            "plot_detail": plot_detail_dict,
            "plot_name": "trsv_error_0,1",
            "plot_prefix": "v100_",
            "label_prefix": "TRSV ",
            "conv_func": lambda sz, error: error,
            "xlabel": "Number of rows",
            "ylabel": "Relative error",
            "yscale": "log",
        },
        {
            #"file": "./results/20210526_1131_a100_trsv_time_ms.csv",
            #"file": "./results/20210526_1148_a100_trsv_time_ms.csv",
            #"file": "./results/20210526_1833_a100_trsv_time_ms.csv",
            "file": "./results/20210526_2025_a100_trsv_time_ms.csv",
            "header_trans": h_dict_trsv_runtime,
            "plot_order": plot_order_flops,
            "plot_detail": plot_detail_dict,
            "plot_name": "trsv_flops",
            "plot_prefix": "a100_",
            "label_prefix": "TRSV ",
            "conv_func": trsv_compute_flop,
            "xlabel": "Number of rows",
            "ylabel": "GFLOP/s",
            "yscale": "linear",
            #"xlim": {"left": None, "right": None,},
            #"ylim": {"bottom": 0, "top": 225,},
        },
        {
            #"file": "./results/20210526_1131_a100_trsv_error.csv",
            #"file": "./results/20210526_1148_a100_trsv_error.csv",
            #"file": "./results/20210526_1833_a100_trsv_error.csv",
            "file": "./results/20210526_2025_a100_trsv_error.csv",
            "header_trans": h_dict_trsv_error,
            "plot_order": plot_order_error,
            "plot_detail": plot_detail_dict,
            "plot_name": "trsv_error",
            "plot_prefix": "a100_",
            "label_prefix": "TRSV ",
            "conv_func": lambda sz, error: error,
            "xlabel": "Number of rows",
            "ylabel": "Relative error",
            "yscale": "log",
        },
        {
            #"file": "./results/20210430_2000_v100_dot.csv",
            #"file": "./results/20210505_1300_v100_dot_32xSM.csv",
            #"file": "./results/20210506_1600_v100_dot_-1,1.csv",
            #"file": "./results/20210506_1800_v100_dot_0,1.csv",
            "file": "./results/20210524_1742_v100_dot.csv",
            "header_trans": h_dict_dot_runtime,
            "plot_order": plot_order_flops,
            "plot_detail": plot_detail_dict,
            "plot_name": "dot_flops",
            "plot_prefix": "v100_",
            "label_prefix": "DOT ",
            "conv_func": dot_compute_flop,
            "xlabel": "Vector size",
            "ylabel": "GFLOP/s",
            "yscale": "linear",
            "xlim": {"left": None, "right": None,},
            "ylim": {"bottom": 0, "top": None,},
            #"ylim": {"bottom": 0, "top": 225,},
        },
        {
            #"file": "./results/20210430_2000_v100_dot.csv",
            #"file": "./results/20210505_1300_v100_dot_32xSM.csv",
            #"file": "./results/20210506_1600_v100_dot_-1,1.csv",
            #"file": "./results/20210506_1800_v100_dot_0,1.csv",
            #"file": "./results/20210506_1930_v100_dot_error_avg_-1,1.csv",
            "file": "./results/20210524_1742_v100_dot_error.csv",
            "header_trans": h_dict_dot_error,
            "plot_order": plot_order_error,
            "plot_detail": plot_detail_dict,
            "plot_name": "dot_error_avg_-1,1",
            "plot_prefix": "v100_",
            "label_prefix": "DOT ",
            "conv_func": lambda sz, error: error,
            "xlabel": "Vector size",
            "ylabel": "Relative error",
            "yscale": "log",
        },
        {
            #"file": "./results/2021020210512_0843_v100_dot_error_avg_0,1.csv",
            "file": "./results/20210829_1839_v100_dot_error_0,1.csv",
            "header_trans": h_dict_dot_error,
            "plot_order": plot_order_error,
            "plot_detail": plot_detail_dict,
            "plot_name": "dot_error_avg_0,1",
            "plot_prefix": "v100_",
            "label_prefix": "DOT ",
            "conv_func": lambda sz, error: error,
            "xlabel": "Vector size",
            "ylabel": "Relative error",
            "yscale": "log",
        },
        {
            #"file": "./results/20210526_1201_a100_dot.csv",
            "file": "./results/20210527_1125_a100_dot_-1,1.csv",
            "header_trans": h_dict_dot_runtime,
            "plot_order": plot_order_flops,
            "plot_detail": plot_detail_dict,
            "plot_name": "dot_flops",
            "plot_prefix": "a100_",
            "label_prefix": "DOT ",
            "conv_func": dot_compute_flop,
            "xlabel": "Vector size",
            "ylabel": "GFLOP/s",
            "yscale": "linear",
            "xlim": {"left": None, "right": None,},
            "ylim": {"bottom": 0, "top": None,},
        },
        {
            #"file": "./results/20210526_1201_a100_dot_error.csv",
            #"file": "./results/20210527_1125_a100_dot_-1,1.csv",
            "file": "./results/20210830_1230_a100_dot_error_-1,1.csv",
            "header_trans": h_dict_dot_error,
            "plot_order": plot_order_error,
            "plot_detail": plot_detail_dict,
            "plot_name": "dot_error_avg_-1,1",
            "plot_prefix": "a100_",
            "label_prefix": "DOT ",
            "conv_func": lambda sz, error: error,
            "xlabel": "Vector size",
            "ylabel": "Relative error",
            "yscale": "log",
        },
        {
            #"file": "./results/2021020210512_0843_v100_dot_error_avg_0,1.csv",
            "file": "./results/20210526_2058_a100_dot_0,1_error.csv",
            "header_trans": h_dict_dot_error,
            "plot_order": plot_order_error,
            "plot_detail": plot_detail_dict,
            "plot_name": "dot_error_avg_0,1",
            "plot_prefix": "a100_",
            "label_prefix": "DOT ",
            "conv_func": lambda sz, error: error,
            "xlabel": "Vector size",
            "ylabel": "Relative error",
            "yscale": "log",
        },
        {
            "file": "./results/20210524_1830_v100_trsv_progress_time_ms.csv",
            "header_trans": {
                "size" : "Num rows",
                "fp64_1": "TRSV multi-kernel fp64",
                "fp64_2": "TRSV single kernel fp64",
                "fp64_3": "TRSV fp64",
                "cublas_fp64": "CUBLAS TRSV fp64",
                },
            "plot_order": ["fp64_1", "fp64_2", "fp64_3", "cublas_fp64"],
            "plot_detail": {
                "fp64_1": {
                    "label": "fp64 multi-kernel",
                    "color": myred,
                    "marker": "",
                    "zorder": 3.1,
                    },
                "fp64_2": {
                    "label": "fp64 single kernel",
                    "color": mygreen,
                    "marker": "",
                    "zorder": 3.2,
                    },
                "fp64_3": {
                    "label": "fp64 inverse L",
                    "color": myblue,
                    "marker": "",
                    "zorder": 3.3,
                    },
                "cublas_fp64": {
                    "label": "cuBLAS fp64",
                    "color": myorange,
                    "marker": "",
                    "zorder": 3.4,
                    },
                },
            "plot_name": "trsv_progress_flops",
            "plot_prefix": "v100_",
            "label_prefix": "TRSV ",
            "conv_func": trsv_compute_flop,
            "xlabel": "Number of rows",
            "ylabel": "GFLOP/s",
            "yscale": "linear",
            #"xlim": {"left": None, "right": None,},
            #"ylim": {"bottom": 0, "top": 225,},
        },
        #{
        #    #"file": "./results/20210430_2000_v100_dot.csv",
        #    #"file": "./results/20210505_1300_v100_dot_32xSM.csv",
        #    "file": "./results/20210506_1800_v100_dot_0,1.csv",
        #    "header_trans": h_dict_dot_error,
        #    "plot_order": plot_order_error,
        #    "plot_detail": plot_detail_dict,
        #    "plot_name": "dot_error_0,1",
        #    "plot_prefix": "v100_",
        #    "label_prefix": "DOT ",
        #    "conv_func": lambda sz, error: error,
        #    "xlabel": "Vector size",
        #    "ylabel": "Relative error",
        #    "yscale": "log",
        #},
        #{
        #    #"file": "./results/20210430_2000_v100_dot.csv",
        #    #"file": "./results/20210505_1300_v100_dot_32xSM.csv",
        #    #"file": "./results/20210506_1600_v100_dot_-1,1.csv",
        #    #"file": "./results/20210506_1800_v100_dot_0,1.csv",
        #    "file": "./results/20210524_1742_v100_dot.csv",
        #    "header_trans": h_dict_dot_error,
        #    "plot_order": plot_order_error,
        #    "plot_detail": plot_detail_dict,
        #    "plot_name": "dot_error_-1,1",
        #    "plot_prefix": "v100_",
        #    "label_prefix": "DOT ",
        #    "conv_func": lambda sz, error: error,
        #    "xlabel": "Vector size",
        #    "ylabel": "Relative error",
        #    "yscale": "log",
        #},
    ]


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

    for plot_info in plot_dict_list:
        data, i_dict = read_csv(plot_info["header_trans"], plot_info["file"])

        plot_data = {}
        plot_data["size"] = []
        
        for line in data:
            if (len(line) < 2):
                break;
            cur_size = int(line[i_dict["size"]])
            plot_data["size"].append(cur_size)
            
            for name, info_dict in plot_info["plot_detail"].items():
                data = float(line[i_dict[name]])
                if name not in plot_data:
                    plot_data[name] = []
                plot_data[name].append(plot_info["conv_func"](cur_size, data))

        fig, ax = create_fig_ax()
        ax.set_yscale(plot_info["yscale"])

        ax.set_xlabel(plot_info["xlabel"], fontsize=LabelFontSize)
        ax.set_ylabel(plot_info["ylabel"], fontsize=LabelFontSize)
        marker_start = 0
        marker_every = int(len(plot_data["size"]) / TotalMarkerPerPlot)
        marker_diff = int(marker_every / len(plot_info["plot_order"]))
        for name in plot_info["plot_order"]:
            info = plot_info["plot_detail"][name]
            ax.plot(plot_data["size"], plot_data[name], label=plot_info["label_prefix"]+info["label"],
                    marker=info["marker"], color=info["color"], linewidth=PlotLineWidth,
                    markersize=MarkerSize, zorder=info["zorder"], markevery=(marker_start, marker_every))
            marker_start = marker_start + marker_diff
        if "xlim" in plot_info:
            ax.set_xlim(**plot_info["xlim"])
        if "ylim" in plot_info:
            ax.set_ylim(**plot_info["ylim"])
        
        ax.tick_params(axis='x', labelsize=AxisTickSize)
        ax.tick_params(axis='y', labelsize=AxisTickSize)
        
        ax.legend(loc="best", fontsize=LabelFontSize)
        #ax.legend(loc="lower right")
        plot_figure(fig, plot_info["plot_name"], plot_info["plot_prefix"])
