## Pre-trained Iris models for structured datasets
This is a code base for the paper ['Pre-training Summarization Models of Structured Datasets for Cardinality Estimation'](http://yao.lu/iris.pdf) accepted to PVLDB15.  We show a version in Python for ease of reading and algorithm development; however, this version is not as performant as our C++ version in terms of build time and query time shown. For more details, please refer to the paper. 

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
Loading ../dataset_public/test/demo_query.txt with 1 sets.
Storage budget 50.625KB, max atom budget 2.0KB, sample size 128 rows
Extracting embedding weights from pre-trained model (can be done once offline)..
Computing quantizations..
Reading data..
Running CORDs..
Building summaries..
        Summary for columnset SHIPINSTRUCT,SHIPMODE     w/ #DV 28,      corr. score 1.0 built using Sparse
        Summary for columnset LINENUMBER,TAX            w/ #DV 63,      corr. score 1.0 built using Sparse
        Summary for columnset LINENUMBER,RETURNFLAG     w/ #DV 21,      corr. score 1.0 built using Sparse
        Summary for columnset LINENUMBER,LINESTATUS     w/ #DV 14,      corr. score 1.0 built using Sparse
        Summary for columnset LINENUMBER,SHIPINSTRUCT   w/ #DV 28,      corr. score 1.0 built using Sparse
        Summary for columnset LINENUMBER,SHIPMODE       w/ #DV 49,      corr. score 1.0 built using Sparse
        Summary for columnset DISCOUNT,RETURNFLAG       w/ #DV 33,      corr. score 1.0 built using Sparse
        Summary for columnset DISCOUNT,LINESTATUS       w/ #DV 22,      corr. score 1.0 built using Sparse
        Summary for columnset DISCOUNT,SHIPINSTRUCT     w/ #DV 44,      corr. score 1.0 built using Sparse
        Summary for columnset TAX,RETURNFLAG            w/ #DV 27,      corr. score 1.0 built using Sparse
        Summary for columnset TAX,LINESTATUS            w/ #DV 18,      corr. score 1.0 built using Sparse
        Summary for columnset TAX,SHIPINSTRUCT          w/ #DV 36,      corr. score 1.0 built using Sparse
        Summary for columnset TAX,SHIPMODE              w/ #DV 63,      corr. score 1.0 built using Sparse
        Summary for columnset RETURNFLAG,LINESTATUS     w/ #DV 4,       corr. score 1.0 built using Sparse
        Summary for columnset RETURNFLAG,SHIPINSTRUCT   w/ #DV 12,      corr. score 1.0 built using Sparse
        Summary for columnset RETURNFLAG,SHIPMODE       w/ #DV 21,      corr. score 1.0 built using Sparse
        Summary for columnset LINESTATUS,SHIPINSTRUCT   w/ #DV 8,       corr. score 1.0 built using Sparse
        Summary for columnset LINESTATUS,SHIPMODE       w/ #DV 14,      corr. score 1.0 built using Sparse
        Summary for columnset QUANTITY,SHIPINSTRUCT     w/ #DV 200,     corr. score 0.00785125  built using Sparse
        Summary for columnset QUANTITY,RETURNFLAG       w/ #DV 150,     corr. score 0.00481957  built using Sparse
        Summary for columnset QUANTITY,LINESTATUS       w/ #DV 100,     corr. score 0.00398584  built using Sparse
        Summary for columnset DISCOUNT,SHIPMODE         w/ #DV 77,      corr. score 0.00178526  built using Sparse
        Summary for columnset DISCOUNT,TAX              w/ #DV 99,      corr. score 0.00172016  built using Sparse
        Summary for columnset LINENUMBER,DISCOUNT       w/ #DV 77,      corr. score 0.00128733  built using Sparse
        Summary for columnset SHIPDATE,RECEIPTDATE      w/ #DV 4853,    corr. score 0.835495    built using Iris
        Summary for columnset SHIPDATE,COMMITDATE       w/ #DV 4969,    corr. score 0.594332    built using Iris
        Summary for columnset COMMITDATE,RECEIPTDATE    w/ #DV 4951,    corr. score 0.584352    built using Iris
        Summary for columnset QUANTITY,SHIPMODE         w/ #DV 350,     corr. score 0.0912474   built using Iris
        Summary for columnset LINENUMBER,QUANTITY       w/ #DV 348,     corr. score 0.0909617   built using Iris
        Summary for columnset QUANTITY,TAX              w/ #DV 450,     corr. score 0.0574449   built using Iris
        Summary for columnset QUANTITY,DISCOUNT         w/ #DV 549,     corr. score 0.0371421   built using Hist
        Summary for columnset QUANTITY,SHIPDATE         w/ #DV 4902,    corr. score 0.00326078  built using Hist
        Summary for columnset QUANTITY,RECEIPTDATE      w/ #DV 4911,    corr. score 0.00315985  built using Hist
        Summary for columnset SUPPKEY,QUANTITY          w/ #DV 4971,    corr. score 0.00314107  built using Hist
        Summary for columnset QUANTITY,COMMITDATE       w/ #DV 4912,    corr. score 0.0029301   built using Hist
        Summary for columnset SUPPKEY,COMMITDATE        w/ #DV 4998,    corr. score 0.00289752  built using Hist
        Summary for columnset SUPPKEY,RECEIPTDATE       w/ #DV 5000,    corr. score 0.00273831  built using Hist
        Summary for columnset SUPPKEY,SHIPDATE          w/ #DV 5000,    corr. score 0.00268218  built using Hist
        Summary for columnset RETURNFLAG,SHIPDATE       w/ #DV 2640,    corr. score 0.0 built using Hist
        Summary for columnset RETURNFLAG,COMMITDATE     w/ #DV 2655,    corr. score 0.0 built using Hist
        Summary for columnset RETURNFLAG,RECEIPTDATE    w/ #DV 2657,    corr. score 0.0 built using Hist
        Summary for columnset LINESTATUS,SHIPDATE       w/ #DV 2156,    corr. score 0.0 built using Hist
        Summary for columnset LINESTATUS,COMMITDATE     w/ #DV 2158,    corr. score 0.0 built using Hist
        Summary for columnset LINESTATUS,RECEIPTDATE    w/ #DV 2165,    corr. score 0.0 built using Hist
        Summary for columnset TAX,RECEIPTDATE           w/ #DV 4473,    corr. score 0.0 built using Hist
        Summary for columnset DISCOUNT,SHIPDATE         w/ #DV 4571,    corr. score 0.0 built using Hist
        Summary for columnset DISCOUNT,COMMITDATE       w/ #DV 4551,    corr. score 0.0 built using Hist
Storage budget: 50.625KB, total used size: 50.078125KB
Base 12, Sparse 24, Hist 17, Iris 6
------------------------------
Computing cardinality estimates..
tmp/cords.log Evaluated 1000 queries
GMQ:1.6757096333449133
95th:4.718244339860523
GMQ low:1.9134749252517456
95th low:5.371967298703262
GMQ med:1.5549995144462376
95th med:3.7747551548335596
GMQ high:1.2341526845114035
95th high:2.0142004005120158
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
### Run the pre-training
To run the pre-training, besides all the testing datasets, download the pre-processed training datasets from the link below and unzip to the root folder. Run the following:
```
cd src/
python run_pretrain.py --nt 2048 --nr 128 --ngpus 4 --model_fnm model_name
```
### Useful command line arguments
`parameters.py` lists all the command line arguments used. We show a few useful ones here:
- `input_fnm`: File specifying the dataset and test test queries. 
- `model_fnm`: The model name used for summarization and CE, and the output model name in pre-training.
- `nusecpp`: If using cpp modules to speed up some parts of the Python code.
- `nr`: The output embedding size for each dataset (\eta in the paper). 
- `storage`: The storage budget for bag of summaries. It is a multiplier of the storage budget used in SQL Server (4KB/column). 
- `max_atom_budget`: Maximum size for an individual summary. 
- `nt`: The input quantization budget (resolution, \ell in the paper).
- `ngpus`: The number of GPUs used in pre-training.
## Download links from Google Drive
- Training datasets [link](https://drive.google.com/file/d/1-S8lkyhOcurUd1BuV6PJekPcSToSyFEo/view?usp=sharing)
- Testing dataset - TPCH-Lineitem [link](https://drive.google.com/file/d/11Xnrn9n4c4RSHuNjKk-ILw41nJ4TMsws/view?usp=sharing)
- Testing dataset - DMV [link](https://drive.google.com/file/d/11U04XtCQZeK5ClLtnTRNsfaESn0fX5LQ/view?usp=sharing)
- Testing dataset - Airline-OnTime [link](https://drive.google.com/file/d/11OPmwHzVxAFLxL2dFnSSKE9iL_lkeXPH/view?usp=sharing)
- Testing dataset - IMDB-CastInfo [link](https://drive.google.com/file/d/11SBnarUKq_zxVVMpMEbpKXCpsZIjnl6b/view?usp=sharing)
- Testing dataset - Poker [link](https://drive.google.com/file/d/11YcZIWRQjOIhOzyYC07PVWwDN_iag-G6/view?usp=sharing)
- Alternative pre-trained models [link](https://drive.google.com/file/d/11ZUZJvwk4wQ-57RZaQ9U37xfd_kAc9qb/view?usp=sharing)
