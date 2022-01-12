# Iris Demo
This is a demo in Python for the paper ['Pre-training Summarization Models of Structured Datasets for Cardinality Estimation'](http://yao.lu/iris/pdf) accepted to PVLDB15.  It showcases (1) using a pre-trained model together with other techniques discussed in the paper to summarize a new dataset (TPCH-LineItem) with a storage budget (60KB, or 4KB per column) that matches the statistics in a production database, and (2) estimating cardinality using the summaries. A single CPU thread is used for both summarization and query answering. For more details, please refer to the paper submission. We are showing a version in Python for ease of reading, however, this version is not as performant as our C++ version in terms of build time and query time. We will update the repo to include code to train the summarization models.

# Tested environment: 
- Ubuntu 18.04
- GCC 7.5.0
- Python 3.7.6
- Swig 3.0.12
- Tensorflow 1.13.1
- Keras 2.3.1

# To run the demo: 
'make
'python run_demo.py

## Console outputs in the tested environment:
Using TensorFlow backend.
Read base per-column histograms..
Loading data/demo_query.txt with 1 sets.
Extracting embedding weights from pre-trained model..
Computing quantizations..
Reading data..
Running CORDs..
Building summaries..
        Summary for columnset SHIPDATE,RECEIPTDATE      w/ #DV 4853,    corr. score 0.832266    built using Iris
        Summary for columnset SHIPDATE,COMMITDATE       w/ #DV 4969,    corr. score 0.675663    built using Iris
        Summary for columnset COMMITDATE,RECEIPTDATE    w/ #DV 4951,    corr. score 0.667107    built using Iris
        Summary for columnset LINESTATUS,SHIPDATE       w/ #DV 2156,    corr. score 0.500365    built using Hist
        Summary for columnset LINESTATUS,RECEIPTDATE    w/ #DV 2165,    corr. score 0.499858    built using Hist
        Summary for columnset LINESTATUS,COMMITDATE     w/ #DV 2158,    corr. score 0.498066    built using Hist
        Summary for columnset SUPPKEY,LINESTATUS        w/ #DV 4414,    corr. score 0.450434    built using Hist
        Summary for columnset RETURNFLAG,LINESTATUS     w/ #DV 4,       corr. score 0.367654    built using Sparse
        Summary for columnset RETURNFLAG,SHIPDATE       w/ #DV 2640,    corr. score 0.322637    built using Hist
        Summary for columnset RETURNFLAG,RECEIPTDATE    w/ #DV 2657,    corr. score 0.321639    built using Hist
        Summary for columnset RETURNFLAG,COMMITDATE     w/ #DV 2655,    corr. score 0.319705    built using Hist
        Summary for columnset SUPPKEY,RETURNFLAG        w/ #DV 4542,    corr. score 0.249295    built using Hist
        Summary for columnset COMMITDATE,SHIPINSTRUCT   w/ #DV 3887,    corr. score 0.1755      built using Hist
        Summary for columnset RECEIPTDATE,SHIPINSTRUCT  w/ #DV 3928,    corr. score 0.17545     built using Hist
        Summary for columnset SHIPDATE,SHIPINSTRUCT     w/ #DV 3938,    corr. score 0.175386    built using Hist
        Summary for columnset SUPPKEY,SHIPINSTRUCT      w/ #DV 4671,    corr. score 0.175204    built using Hist
        Summary for columnset QUANTITY,SHIPMODE         w/ #DV 350,     corr. score 0.0586023   built using Hist
        Summary for columnset SUPPKEY,SHIPMODE          w/ #DV 4816,    corr. score 0.0583625   built using Hist
        Summary for columnset LINENUMBER,QUANTITY       w/ #DV 348,     corr. score 0.0583245   built using Hist
        Summary for columnset COMMITDATE,SHIPMODE       w/ #DV 4312,    corr. score 0.0582801   built using Hist
        Summary for columnset SHIPDATE,SHIPMODE         w/ #DV 4343,    corr. score 0.0581721   built using Hist
        Summary for columnset RECEIPTDATE,SHIPMODE      w/ #DV 4359,    corr. score 0.0580574   built using Hist
        Summary for columnset SUPPKEY,LINENUMBER        w/ #DV 4798,    corr. score 0.057926    built using Hist
        Summary for columnset LINENUMBER,RECEIPTDATE    w/ #DV 4205,    corr. score 0.0577431   built using Hist
        Summary for columnset LINENUMBER,SHIPDATE       w/ #DV 4215,    corr. score 0.0576707   built using Hist
        Summary for columnset LINENUMBER,COMMITDATE     w/ #DV 4191,    corr. score 0.0576335   built using Hist
        Summary for columnset QUANTITY,TAX              w/ #DV 450,     corr. score 0.0245001   built using Hist
        Summary for columnset SUPPKEY,TAX               w/ #DV 4874,    corr. score 0.0241577   built using Hist
        Summary for columnset TAX,SHIPDATE              w/ #DV 4504,    corr. score 0.0239146   built using Hist
Storage budget: 60KB, total used size: 58.296875KB
------------------------------
Computing cardinality estimates..
1000 queries evaluated.
        Iris            GMQ:1.64, 95th:4.34
        For comparison, the following baseline results are pre-computed.
        Sampling        GMQ:3.02, 95th:109.98
        xAVI            GMQ:2.94, 95th:21.00
        LM-             GMQ:2.21, 95th:8.08
        MSCN            GMQ:3.62, 95th:52.0