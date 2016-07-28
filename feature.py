import sys, os
import fastq

MAX_LEN = 1000
READ_TO_SKIP = 1000
ALL_BASES = ("A", "T", "C", "G");

class Feature:
    def __init__(self, filename, sample_limit=1000000):
        self.sample_limit = sample_limit
        self.filename = filename
        self.percents = {}
        self.total_num = [0 for x in xrange(MAX_LEN)]
        for base in ALL_BASES:
            self.base_counts[base] = [0 for x in xrange(MAX_LEN)]
            self.percents[base] = [0.0 for x in xrange(MAX_LEN)]

    def stat_read(self, read):
        seq = read[1]
        seqlen = len(seq)
        for i in xrange(seqlen):
            self.total_num[i] += 1
            b = seq[i]
            if b in ALL_BASES:
                self.base_counts[b][i] += 1

    def stat(self):
        reader = fastq.Reader(filename)
        stat_reads_num = 0
        skipped_reads = []
        #sample up to maxSample reads for stat
        while True:
            read = reader.nextRead()
            if read==None:
                break
            self.readCount += 1
            # here we skip the first 1000 reads because usually they are usually not stable
            if self.readCount < READ_TO_SKIP:
                skipped_reads.append(read)
                continue
            stat_reads_num += 1
            if stat_reads_num > self.sample_limit and self.sample_limit>0:
                break
            self.stat_read(read)

        # if the fq file is too small, then we stat the skipped reads again
        if stat_reads_num < READ_TO_SKIP:
            for read in skipped_reads:
                self.stat_read(read)