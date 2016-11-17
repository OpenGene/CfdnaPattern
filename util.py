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