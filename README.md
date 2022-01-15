## Pre-trained Iris models for structured datasets
This is a code base for the paper ['Pre-training Summarization Models of Structured Datasets for Cardinality Estimation'](http://yao.lu/iris.pdf) accepted to PVLDB15.  We show a version in Python for ease of reading and algorithm development.

## Tested environment: 
- Ubuntu 18.04
- GCC 7.5.0
- Python 3.7.6
- Swig 3.0.12
- Tensorflow 1.13.1
- Keras 2.3.1

## Run demo for summary and CE: 
We showcase a simple demo: (1) using a pre-trained model together with other techniques discussed in the paper to summarize a new dataset (TPCH-LineItem, sampled) with a storage budget (60KB, or 4KB per column) that matches the statistics in a production database, and (2) estimating cardinality using the summaries. A single CPU thread is used for both summarization and query answering.

To run the demo, 
```
cd src/
make
python run_summary_CE.py
```

### Console outputs in the tested environment:
```
Using TensorFlow backend.
--------------Run Iris demo----------------
Loading ../dataset_public/test/demo_query.txt with 1 sets.
Storage budget 60KB, max atom budget 2.0KB, sample size 128 rows
Storage budget exlucding overhead 50.625KB
Extracting embedding weights from pre-trained model (can be cached offline)..
Computing quantizations..
Reading data..
--------------Part I-----------------
Running CORDs..
Building summaries using pre-trained ../results/Irisv0.1-nml2Mnt2048nr128nm0.5.model..
        Columnset SHIPINSTRUCT,SHIPMODE         w/ #DV 28,      corr. score 1.0 built using Sparse
        Columnset LINENUMBER,TAX                w/ #DV 63,      corr. score 1.0 built using Sparse
        Columnset LINENUMBER,RETURNFLAG         w/ #DV 21,      corr. score 1.0 built using Sparse
        Columnset LINENUMBER,LINESTATUS         w/ #DV 14,      corr. score 1.0 built using Sparse
        Columnset LINENUMBER,SHIPINSTRUCT       w/ #DV 28,      corr. score 1.0 built using Sparse
        Columnset LINENUMBER,SHIPMODE           w/ #DV 49,      corr. score 1.0 built using Sparse
        Columnset DISCOUNT,RETURNFLAG           w/ #DV 33,      corr. score 1.0 built using Sparse
        Columnset DISCOUNT,LINESTATUS           w/ #DV 22,      corr. score 1.0 built using Sparse
        Columnset DISCOUNT,SHIPINSTRUCT         w/ #DV 44,      corr. score 1.0 built using Sparse
        Columnset TAX,RETURNFLAG                w/ #DV 27,      corr. score 1.0 built using Sparse
        Columnset TAX,LINESTATUS                w/ #DV 18,      corr. score 1.0 built using Sparse
        Columnset TAX,SHIPINSTRUCT              w/ #DV 36,      corr. score 1.0 built using Sparse
        Columnset TAX,SHIPMODE                  w/ #DV 63,      corr. score 1.0 built using Sparse
        Columnset RETURNFLAG,LINESTATUS         w/ #DV 4,       corr. score 1.0 built using Sparse
        Columnset RETURNFLAG,SHIPINSTRUCT       w/ #DV 12,      corr. score 1.0 built using Sparse
        Columnset RETURNFLAG,SHIPMODE           w/ #DV 21,      corr. score 1.0 built using Sparse
        Columnset LINESTATUS,SHIPINSTRUCT       w/ #DV 8,       corr. score 1.0 built using Sparse
        Columnset LINESTATUS,SHIPMODE           w/ #DV 14,      corr. score 1.0 built using Sparse
        Columnset QUANTITY,SHIPINSTRUCT         w/ #DV 200,     corr. score 0.00785125  built using Sparse
        Columnset QUANTITY,RETURNFLAG           w/ #DV 150,     corr. score 0.00481957  built using Sparse
        Columnset QUANTITY,LINESTATUS           w/ #DV 100,     corr. score 0.00398584  built using Sparse
        Columnset DISCOUNT,SHIPMODE             w/ #DV 77,      corr. score 0.00178526  built using Sparse
        Columnset DISCOUNT,TAX                  w/ #DV 99,      corr. score 0.00172016  built using Sparse
        Columnset LINENUMBER,DISCOUNT           w/ #DV 77,      corr. score 0.00128733  built using Sparse
        Columnset SHIPDATE,RECEIPTDATE          w/ #DV 4853,    corr. score 0.835495    built using Iris
        Columnset SHIPDATE,COMMITDATE           w/ #DV 4969,    corr. score 0.594332    built using Iris
        Columnset COMMITDATE,RECEIPTDATE        w/ #DV 4951,    corr. score 0.584352    built using Iris
        Columnset QUANTITY,SHIPMODE             w/ #DV 350,     corr. score 0.0912474   built using Iris
        Columnset LINENUMBER,QUANTITY           w/ #DV 348,     corr. score 0.0909617   built using Iris
        Columnset QUANTITY,TAX                  w/ #DV 450,     corr. score 0.0574449   built using Iris
        Columnset QUANTITY,DISCOUNT             w/ #DV 549,     corr. score 0.0371421   built using Hist
        Columnset QUANTITY,SHIPDATE             w/ #DV 4902,    corr. score 0.00326078  built using Hist
        Columnset QUANTITY,RECEIPTDATE          w/ #DV 4911,    corr. score 0.00315985  built using Hist
        Columnset SUPPKEY,QUANTITY              w/ #DV 4971,    corr. score 0.00314107  built using Hist
        Columnset QUANTITY,COMMITDATE           w/ #DV 4912,    corr. score 0.0029301   built using Hist
        Columnset SUPPKEY,COMMITDATE            w/ #DV 4998,    corr. score 0.00289752  built using Hist
        Columnset SUPPKEY,RECEIPTDATE           w/ #DV 5000,    corr. score 0.00273831  built using Hist
        Columnset SUPPKEY,SHIPDATE              w/ #DV 5000,    corr. score 0.00268218  built using Hist
        Columnset RETURNFLAG,SHIPDATE           w/ #DV 2640,    corr. score 0.0 built using Hist
        Columnset RETURNFLAG,COMMITDATE         w/ #DV 2655,    corr. score 0.0 built using Hist
        Columnset RETURNFLAG,RECEIPTDATE        w/ #DV 2657,    corr. score 0.0 built using Hist
        Columnset LINESTATUS,SHIPDATE           w/ #DV 2156,    corr. score 0.0 built using Hist
        Columnset LINESTATUS,COMMITDATE         w/ #DV 2158,    corr. score 0.0 built using Hist
        Columnset LINESTATUS,RECEIPTDATE        w/ #DV 2165,    corr. score 0.0 built using Hist
        Columnset TAX,RECEIPTDATE               w/ #DV 4473,    corr. score 0.0 built using Hist
        Columnset DISCOUNT,SHIPDATE             w/ #DV 4571,    corr. score 0.0 built using Hist
        Columnset DISCOUNT,COMMITDATE           w/ #DV 4551,    corr. score 0.0 built using Hist
Storage budget: 50.625KB, total used size: 50.078125KB
Base 12, Sparse 24, Hist 17, Iris 6
--------------Part II-----------------
Computing cardinality estimates..
test/demo_query.txt Evaluated 1000 queries
        Iris            GMQ:1.68, 95th:4.72
        For comparison, the following baseline results are pre-computed.
        Sampling        GMQ:3.02, 95th:109.98
        xAVI            GMQ:2.94, 95th:21.00
        LM-             GMQ:2.21, 95th:8.08
        MSCN            GMQ:3.62, 95th:52.0
```
### Run summary and CE on full test datasets in the paper: 
Download the test dataset from the link below. Unzip to the root folder. Run the following:
```
cd src/
python run_summary_CE.py --input_fnm test/full-lineitem.txt 
```
### Run the pre-training:
To run the pre-training, besides all the testing datasets, download the pre-processed training datasets from the link below and unzip to the root folder. Run the following:
```
cd src/
python run_pretrain.py --nt 2048 --nr 128 --ngpus 4 --model_fnm model_name
```
### Useful command line arguments:
`src/parameters.py` lists all the command line arguments used. We show a few useful ones here:
- `input_fnm`: File specifying the dataset and test test queries. 
- `model_fnm`: The model name used for summarization and CE, and the output model name in pre-training.
- `nusecpp`: If using cpp modules to speed up some parts of the Python code.
- `nr`: The output embedding size for each dataset (\eta in the paper). 
- `storage`: The storage budget for bag of summaries. It is a multiplier of the storage budget used in SQL Server (4KB/column). 
- `max_atom_budget`: Maximum size for an individual summary. 
- `nt`: The input quantization budget (resolution, \ell in the paper).
- `ngpus`: The number of GPUs used in pre-training.
## Download links:
- Pre-processed training datasets [link](https://drive.google.com/file/d/1-S8lkyhOcurUd1BuV6PJekPcSToSyFEo/view?usp=sharing)
- Pre-processed testing dataset - TPCH-Lineitem [link](https://drive.google.com/file/d/11Xnrn9n4c4RSHuNjKk-ILw41nJ4TMsws/view?usp=sharing)
- Pre-processed testing dataset - DMV [link](https://drive.google.com/file/d/11U04XtCQZeK5ClLtnTRNsfaESn0fX5LQ/view?usp=sharing)
- Pre-processed testing dataset - Airline-OnTime [link](https://drive.google.com/file/d/11OPmwHzVxAFLxL2dFnSSKE9iL_lkeXPH/view?usp=sharing)
- Pre-processed testing dataset - IMDB-CastInfo [link](https://drive.google.com/file/d/11SBnarUKq_zxVVMpMEbpKXCpsZIjnl6b/view?usp=sharing)
- Pre-processed testing dataset - Poker [link](https://drive.google.com/file/d/11YcZIWRQjOIhOzyYC07PVWwDN_iag-G6/view?usp=sharing)
- Alternative pre-trained models [link](https://drive.google.com/file/d/11ZUZJvwk4wQ-57RZaQ9U37xfd_kAc9qb/view?usp=sharing)
## Remark
Due to license issues, this release version does not contain the C++ code such as SIMD model inference and standalone dataset summarization and is not as performant in terms of building time and CE time shown in our the paper.
