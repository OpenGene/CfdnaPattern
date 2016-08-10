import sys, os
from feature import *

# try to load matplotlib
HAVE_MATPLOTLIB = True

try:
    import matplotlib
    # fix matplotlib DISPLAY issue
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except Exception:
    HAVE_MATPLOTLIB = False

def plot_data_list(wrong_files, wrong_data, figure_dir):
    if not HAVE_MATPLOTLIB:
        print("\nmatplotlib not installed, skip plotting figures for files with wrong predictions")

    print("\nplotting figures for files with wrong predictions...")
    if not os.path.exists(figure_dir):
        try:
            os.mkdir(figure_dir)
        except Exception:
            print("failed to create folder to store figures")
            return
    for i in xrange(len(wrong_files)):
        filename = wrong_files[i]
        f = os.path.join(figure_dir, filename.strip('/').replace("/", "-") + ".png")
        plot_data(wrong_data[i], f, filename[filename.rfind('/')+1:])

def plot_data(data, filename, title):
    # data is packed as values of ATCGATCGATCGATCGATCG
    colors = {'A':'red', 'T':'purple', 'C':'blue', 'G':'green'}
    base_num = len(ALL_BASES)
    cycles = len(data)/base_num
    percents = {}
    for b in xrange(base_num):
        percents[ALL_BASES[b]]=[ 0.0 for c in xrange(cycles)]

    for c in xrange(cycles):
        total = 0
        for b in xrange(base_num):
            total += data[c * base_num + b]
        for b in xrange(base_num):
            percents[ALL_BASES[b]][c] = float(data[c * base_num + b]) / float(total)

    x = range(cycles)
    plt.figure(1)
    plt.title(title)
    plt.xlim(0, cycles)
    max_y = 0.8
    for base in ALL_BASES:
        max_of_base = max(percents[base][0:cycles])
        max_y = max(max_y, max_of_base+0.05)
    plt.ylim(0.0, max_y )
    plt.ylabel('Percents')
    plt.xlabel('Cycle')
    for base in ALL_BASES:
        plt.plot(x, percents[base][0:cycles], color = colors[base], label=base, alpha=0.5)
    plt.legend(loc='upper right', ncol=5)
    plt.savefig(filename)
    plt.close(1)