import sys, os
import fastq
import skip

def is_to_skip(path):
    for item in skip.file_to_skip:
        if item[0] in path and item[1] in path:
            return True
    return False

def get_arg_files():
    files = []
    for f in sys.argv:
        if fastq.is_fastq(f) and os.path.exists(f):
            path = os.path.join(os.getcwd(), f)
            if not is_to_skip(path):
                files.append(path)
    return files

def has_adapter_sequenced(data):
    # work around for skipping the data with 6bp index, sequenced in 8bp index setting
    count = 0
    for i in range(8):
        if data[i]>0.7:
            count += 1
    return count >= 1