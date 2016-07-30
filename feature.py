import sys, os
import fastq

STAT_LEN_LIMIT = 10
READ_TO_SKIP = 1000
ALL_BASES = ("A", "T", "C", "G");

class FeatureExtractor:
    def __init__(self, filename, sample_limit=10000):
        self.sample_limit = sample_limit
        self.filename = filename
        self.base_counts = {}
        self.percents = {}
        self.read_count = 0
        self.stat_len = 0
        self.total_num = [0 for x in xrange(STAT_LEN_LIMIT)]
        for base in ALL_BASES:
            self.base_counts[base] = [0 for x in xrange(STAT_LEN_LIMIT)]
            self.percents[base] = [0.0 for x in xrange(STAT_LEN_LIMIT)]

    def stat_read(self, read):
        seq = read[1]
        seqlen = len(seq)
        for i in xrange(min(seqlen, STAT_LEN_LIMIT)):
            self.total_num[i] += 1
            b = seq[i]
            if b in ALL_BASES:
                self.base_counts[b][i] += 1

    def extract(self):
        reader = fastq.Reader(self.filename)
        stat_reads_num = 0
        skipped_reads = []
        #sample up to maxSample reads for stat
        while True:
            read = reader.nextRead()
            if read==None:
                break
            self.read_count += 1
            # here we skip the first 1000 reads because usually they are usually not stable
            if self.read_count < READ_TO_SKIP:
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

        self.calc_read_len()
        self.calc_percents()

    def calc_read_len(self):
        for pos in xrange(STAT_LEN_LIMIT):
            has_data = False
            for base in ALL_BASES:
                if self.base_counts[base][pos]>0:
                    has_data = True
            if has_data == False:
                self.stat_len = pos
                return
        if has_data:
            self.stat_len = STAT_LEN_LIMIT

    def calc_percents(self):
        #calc percents of each base
        for pos in xrange(self.stat_len):
            total = 0
            for base in ALL_BASES:
                total += self.base_counts[base][pos]
            for base in ALL_BASES:
                self.percents[base][pos] = float(self.base_counts[base][pos])/float(total)

    def feature(self):
        # bad feature
        if self.stat_len < STAT_LEN_LIMIT:
            return None
        #calc percents of each base
        feature_vector = []
        for pos in xrange(self.stat_len):
            total = 0
            for base in ALL_BASES:
                feature_vector.append(self.percents[base][pos])
        return feature_vector
