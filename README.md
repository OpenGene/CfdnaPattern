# CfdnaPattern
Pattern Recognition for Cell-free DNA

# Predict a fastq is cfdna or not
```shell
# predict a single file
python predict.py <singlefastq_file>

# predict files
python predict.py <fastq_file1> <fastq_file2> ... 

# predict files with wildcard
python predict.py *.fq
```

## prediction output
For each file in the command line, this tool will output a line `<prediction>: <filename>`, like
```
cfdna: /fq/160220_NS500713_0040_AHVNG2BGXX/20160220-cfdna-001_S1_R1_001.fastq.gz
cfdna: /fq/160220_NS500713_0040_AHVNG2BGXX/20160220-cfdna-001_S1_R2_001.fastq.gz
not-cfdna: /fq/160220_NS500713_0040_AHVNG2BGXX/20160220-gdna-002_S2_R1_001.fastq.gz
not-cfdna: /fq/160220_NS500713_0040_AHVNG2BGXX/20160220-gdna-002_S2_R2_001.fastq.gz
```
Add `-q` to enable quite mode, in which it will only output:
* a file with name of `cfdna`, but prediction is `not-cfdna`
* a file without name of `cfdna`, but prediction is `cfdna`

# Train a model
This tool has a pre-trained model (`cfdna.model`), which can be used for prediction. But you still can train a model by yourself.
* prepare/link all your fastq files in some folder
* for files from `cfdna`, include `cfdna` (case-insensitive) in the filename, like `20160220-cfdna-015_S15_R1_001.fq`
* for files from `genomic DNA`, include `gdna` (case-insensitive) in the filename, like `20160220-gdna-002_S2_R1_001.fq`
* for files from `FFPE DNA`, include `ffpe` (case-insensitive) in the filename, like `20160123-ffpe-040_S0_R1_001.fq`
* run:
```shell
python train.py /fastq_folder/*.fq
```
Full options:
```shell
python training.py <fastq_files> [options] 

Options:
  --version             show program's version number and exit
  -h, --help            show this help message and exit
  -f FEATURE_FILE, --feature=FEATURE_FILE
                        specify which file to store the extracted features
                        from training set.
  -m MODEL_FILE, --model=MODEL_FILE
                        specify which file to store the built model.
  -a ALGORITHM, --algorithm=ALGORITHM
                        specify which algorithm to use for classfication,
                        candidates are svm/knn, default is knn.
  -c CFDNA_FLAG, --cfdna_flag=CFDNA_FLAG
                        specify the filename flag of cfdna files, separated by
                        semicolon. default is: cfdna
  -o OTHER_FLAG, --other_flag=OTHER_FLAG
                        specify the filename flag of other files, separated by
                        semicolon. default is: gdna;ffpe
  -p PASSES, --passes=PASSES
                        specify how many passes to do training and validating,
                        default is 10.
  -n, --no_cache_check  if the cache file exists, use it without checking the
                        identity with input files
```
