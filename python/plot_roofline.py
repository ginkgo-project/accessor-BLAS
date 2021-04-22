#!/usr/bin/env python3
import os
import csv
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_pdf import PdfPages


plot_folder = "./plots/"


### dictionary to match purpose to CSV header
h_dict = {
        "rows" : "Num Rows",
        #"GOPS": "[GOPs/s]", # Old value, changed to `[GOP/s]`
        "double": "GEMV double",
        "float": "GEMV float",
        "acc": "GEMV Acc<fp32, fp64>",
        }

def read_csv(path=None):
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
myblue    = (0, 0.4470, 0.7410);
myorange  = (0.8500, 0.3250, 0.0980);
myyellow  = (0.9290, 0.6940, 0.1250);
mymagenta = (0.4940, 0.1840, 0.5560);
mygreen   = (0.4660, 0.6740, 0.1880);
mycyan    = (0.3010, 0.7450, 0.9330);
myred     = (0.6350, 0.0780, 0.1840);
myblack   = (0.2500, 0.2500, 0.2500);
mybrown   = (0.6500, 0.1600, 0.1600);

dark_mod = 2
mydarkred     = (0.6350 / dark_mod, 0.0780 / dark_mod, 0.1840 / dark_mod);
mydarkgreen   = (0.4660 / dark_mod, 0.6740 / dark_mod, 0.1880 / dark_mod);
mydarkblue    = (0, 0.4470 / dark_mod, 0.7410 / dark_mod);

### Other globals
LineWidth = 3
MarkerSize = 8



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


def plot_for_all(ax, data, x_key, y_key):
    """
    plots given x and y keys for all precisions of interest on the axis ax.
    """
    markers = ('X', 'P', 'x', '+')
    precs = ("double", "float",  "Ac<3, d, d>", "Ac<3, d, f>")
    colors = (mygreen, myblue, myorange, myyellow)
    labels = ("fp64", "fp32",  "Accessor<fp64, fp64>", "Accessor<fp64, fp32>")
    for i in range(len(precs)):
        ax.plot(data[precs[i]][x_key], data[precs[i]][y_key], label=labels[i],
                marker=markers[i], color=colors[i], linewidth=LineWidth,
                markersize=MarkerSize)
    """ To get / set x- and y-limits:
    ax.set_xlim(0.7070722721781199, 1449.6396483523677)
    ax.set_ylim(148.24516110946269, 24024.62127583265)
    xl, xr = ax.get_xlim()
    yl, yr = ax.get_ylim()
    print("xlim: ({}, {}); ylim: ({}, {})".format(xl, xr, yl, yr));
    """


if __name__ == "__main__":
    bw_color = myblack
    fp64_color = mydarkgreen
    fp32_color = mydarkblue

    # Change to the directory where the script is placed
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Make sure the plot folder exists
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    data, i_dict = read_csv("./v100_gemv.csv")

    # Generate data for plotting for all available precisions
    plot_data = {}
    use_global_elms = False
    # When available, use "elems"

    num_rows = []
    double_flop = []
    float_flop = []
    acc_flop = []
    for line in data:
        if (len(line) < 2):
            break;
        rows = int(line[i_dict["rows"]])
        dbl = float(line[i_dict["double"]])
        flt = float(line[i_dict["float"]])
        acc = float(line[i_dict["acc"]])

        mflops = rows * rows / (1000 * 1000)
        num_rows.append(rows)
        double_flop.append(mflops / dbl)
        float_flop.append(mflops / flt)
        acc_flop.append(mflops / acc)

    fig, ax = create_fig_ax()

    ax.set_xlabel("Number of rows")
    ax.set_ylabel("GFLOP/s")
    ax.plot(num_rows, double_flop, label="GEMV double",
            marker='', color=myred, linewidth=LineWidth,
            markersize=MarkerSize)
    ax.plot(num_rows, float_flop, label="GEMV float",
            marker='', color=mygreen, linewidth=LineWidth,
            markersize=MarkerSize)
    ax.plot(num_rows, acc_flop, label="GEMV Accessor<fp64, fp32>",
            marker='', color=myblue, linewidth=LineWidth,
            markersize=MarkerSize)
    #ax.legend(loc="best")
    ax.legend(loc="lower right")
    plot_figure(fig, "gemv_flops", "v100")



