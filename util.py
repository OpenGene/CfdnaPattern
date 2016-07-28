import sys, os
import fastq

def get_arg_files():
    files = []
    for f in sys.argv:
        if fastq.is_fastq(f) and os.path.exists(f):
            path = os.path.join(os.getcwd(), f)
            files.append(path)
    return files