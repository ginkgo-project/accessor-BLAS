#!/usr/bin/env python3
import os
import csv
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_pdf import PdfPages


# TODO: Add marker at different points to better distinguish the lines

plot_folder = "./plots/"


h_dict_dot_error = {
        "rnd" : "Random iter",
        "size" : "Vector Size",
        "fp64": "Result DOT fp64",
        "fp32": "Result DOT fp32",
        "acc_mix": "Result DOT Acc<fp64, fp32>",
        "cublas_fp64": "Result CUBLAS DOT fp64",
        "cublas_fp32": "Result CUBLAS DOT fp32",
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

def arithmetic_mean(values):
    result = 0.0
    for i in values:
        result = result + i
    return result / len(values)

def geometric_mean(values):
    result = 1.0
    n = 0
    for i in values:
        if i != 0.0:
            result = result * i
            n = n+1
    return result**(1/n)

def median(values):
    values.sort()
    if len(values) % 2 == 1:
        return values[int(len(values)/2)]
    middle_start = int(len(values)/2 - 1)
    return (values[middle_start] + values[middle_start + 1]) / 2


plot_dict_list = [
        {
            "file": "./results/20210830_1608_v100_dot_error_detail_-1,1_median.csv",
            "header_trans": h_dict_dot_error,
            "plot_order": plot_order_error,
            "plot_detail": plot_detail_dict,
            "plot_name": "dot_error_test_-1,1",
            #"mean_method": arithmetic_mean,
            #"mean_method": geometric_mean,
            "mean_method": median,
            "plot_prefix": "v100_",
            "label_prefix": "DOT ",
            "xlabel": "Vector size",
            "ylabel": "Relative error",
            "yscale": "log",
        },
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

        size_to_idx = {}
        next_idx = 0
        size_array = []
        result_data = {}
        reference_name = "fp64"
        
        for line in data:
            if (len(line) < 2):
                break;
            cur_size = int(line[i_dict["size"]])
            if cur_size not in size_to_idx:
                size_array.append(cur_size)
                size_to_idx[cur_size] = next_idx
                next_idx = next_idx + 1
            
            for name in plot_info["plot_order"]:
                data = float(line[i_dict[name]])
                ref_data = float(line[i_dict[reference_name]])
                data = abs(data - ref_data) / abs(ref_data)
                if name not in result_data:
                    result_data[name] = []
                if len(result_data[name]) <= size_to_idx[cur_size]:
                    result_data[name].append([])
                result_data[name][size_to_idx[cur_size]].append(data)

        plot_data = {}
        plot_data["size"] = size_array

        for name, results in result_data.items():
            plot_data[name] = []
            for i in range(len(size_array)):
                plot_data[name].append(plot_info["mean_method"](results[i]))


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
