import sys, os
from feature import *

# try to load matplotlib
HAVE_MATPLOTLIB = True

try:
    import matplotlib
    # fix matplotlib DISPLAY issue
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rc('axes',edgecolor='#AAAAAA', labelcolor='#666666')
    matplotlib.rc('xtick',color='#666666')
    matplotlib.rc('ytick',color='#666666')
except Exception:
    HAVE_MATPLOTLIB = False

def plot_data_list(wrong_files, wrong_data, figure_dir):
    if not HAVE_MATPLOTLIB:
        print("\nmatplotlib not installed, skip plotting figures for files with wrong predictions")
        return

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

    x = range(1, cycles+1)
    plt.figure(1, figsize=(5.5,3), edgecolor='#cccccc')
    plt.title(title[0:title.find('.')], size=10)
    plt.xlim(1, cycles)
    max_y = 0.35
    min_y = 0.15
    for base in ALL_BASES:
        max_of_base = max(percents[base][0:cycles])
        max_y = max(max_y, max_of_base+0.05)
        min_of_base = min(percents[base][0:cycles])
        min_y = min(min_y, min_of_base-0.05)
    plt.ylim(min_y, max_y )
    plt.ylabel('Ratio')
    #plt.xlabel('Cycle')
    for base in ALL_BASES:
        plt.plot(x, percents[base][0:cycles], color = colors[base], label=base, alpha=0.5, linewidth=2, marker='o', markeredgewidth=0.0, markersize=4)
    #plt.legend(loc='upper right', ncol=5)
    plt.savefig(filename)
    plt.close(1)

def plot_benchmark(scores_arr, algorithms_arr, filename):
    colors = ['#FF6600', '#009933', '#2244AA', '#552299', '#11BBDD']
    linestyles = ['-', '--', ':']
    passes = len(scores_arr[0])

    x = range(1, passes+1)
    title = "Benchmark Result"
    plt.figure(1, figsize=(8,8))
    plt.title(title, size=20, color='#333333')
    plt.xlim(1, passes)
    plt.ylim(0.95, 1.001)
    plt.ylabel('Score', size=16, color='#333333')
    plt.xlabel('Validation pass (sorted by score)', size=16, color='#333333')
    for i in xrange(len(scores_arr)):
        plt.plot(x, scores_arr[i], color = colors[i%5], label=algorithms_arr[i], alpha=0.5, linewidth=2, linestyle = linestyles[i%3])
    plt.legend(loc='lower left')
    plt.savefig(filename)
    plt.close(1)