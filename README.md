# CfdnaPattern
Pattern Recognition for Cell-free DNA

# Training
* prepare/link all your fastq files in some folder
* for files from `cfdna`, include `cfdna` (case-insensitive) in the filename, like `20160220-cfdna-015_S15_R1_001.fq`
* for files from `genomic DNA`, include `gdna` (case-insensitive) in the filename, like `20160220-gdna-002_S2_R1_001.fq`
* for files from `FFPE DNA`, include `ffpe` (case-insensitive) in the filename, like `20160123-ffpe-040_S0_R1_001.fq`
* run:
```shell
python train.py /fastq_folder/*.fq
```

# Prediction
```shell
# predict a single file
python predict.py <singlefastq_file>

# predict files
python predict.py <fastq_file1> <fastq_file2> ... 

# predict files with wildcard
python predict.py *.fq
```
