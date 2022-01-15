import optparse
import os

def parse_arg(istest=2):
    parser = optparse.OptionParser()
    
    parser.add_option('--cords_fnm', dest='cords_fnm', default='tmp/cords.log', type='str')
    parser.add_option('--input_fnm', dest='input_fnm', default='test/demo_query.txt', type='str')
    parser.add_option('--data_dir', dest='data_dir', default='../dataset_public', type='str')
    parser.add_option('--output_dir', dest='output_dir', default='../results/', type='str')
    parser.add_option('--model_fnm', dest='model_fnm', default='../results/Irisv0.1-nml2Mnt2048nr128nm0.5.model', type='str')

    parser.add_option('--run_cords', dest="run_cords", help="Run cords or use cache. Default: 1", default=1, type='int')
    parser.add_option('--extract_emb', dest="extract_emb", help="Extract embedding vectors from input model or use cache. Default: 1", default=1, type='int')

    parser.add_option('--input_rate', dest="input_rate", help="Input dataset size. Default: 1.0", default=1, type='float')
    parser.add_option('--sample_size', dest="sample_size", help="Sample Size. Default: 128", default=128, type='int')
    parser.add_option('--nusecpp', dest="nusecpp", help="Use CPP. Default: 0", default=0, type='int')
    parser.add_option('--nusebucket', dest="nusebucket", help="Use prior buckets. Default: ''(empty)", default='', type='str')

    parser.add_option('--nn', dest="nn", help="# FC layers. Default: 0.", default=0, type='int')
    parser.add_option('--nfc', dest="nfc", help="FC layers size. Default: 128.", default=128, type='int')
    parser.add_option('--nr', dest="nr", help="Embedding size. Default: 128.", default=128, type='int')

    parser.add_option('--storage', dest="storage", help="Storage used (X 4KB/col) including overhead", default=1, type='float')
    parser.add_option('--max_atom_budget', dest="max_atom_budget", help="max atom budget (# floats)", default=512, type='int')

    parser.add_option('--nm', dest="nm", help="Training compression rate. Default: 0.5.", default=0.5, type='float')
    parser.add_option('--nt', dest="nt", help="Total cell embedding budget. Default: 2048.", default=2048, type='int')

    parser.add_option('--neb', dest="neb", help="Max column resolution: 128.", default=128, type='int')

    parser.add_option('--nlen', dest="nlen", help="Input length. Default: 2000000.", default=2000000, type='int')
    parser.add_option('--normlen', dest="normlen", help="Norm length. Default: 2000000.", default=2000000, type='int')

    parser.add_option('--mind', dest='mind', help='Minimum sketch dimensions. Defalut: 2', default=2, type='int')
    parser.add_option('--maxd', dest='maxd', help='Maximum sketch dimensions. Defalut: 2', default=2, type='int')
    parser.add_option('--nb', dest="nb", help="Base size. Default: 2.", default=2, type='int')

    # training parameters
    parser.add_option('--nlr', dest="nlr", help="Learning rate. Default: 1e-2.", default=1e-2, type='float')
    parser.add_option('--ngpus', dest="ngpus", help="# GPUs. Default: 1.", default=1, type='int')
    parser.add_option('--nl', dest="nl", help="Training pool size. Default: 30x.", default=30, type='float')
    parser.add_option('--nbat', dest="nbat", help="Batch size. Default: 2048.", default=2048, type='int')
    parser.add_option('--nst', dest="nst", help="# of steps per epoch: 1280.", default=1280, type='int')
    parser.add_option('--nep', dest="nep", help="# of Epoches. Default: 64.", default=64, type='int')
    parser.add_option('--ne', dest="ne", help="Individual cell embedding length. Default: 64.", default=64, type='int')
    parser.add_option('--verbose', dest="nverbose", help="verbose.", default=1, type='int')

    options, args = parser.parse_args()
    options.ncd = options.nt - (int)(options.nt * (1 - options.nm))
    options.isdemo = 0
    if istest == 2 and options.storage == 1 and ('demo_query.txt' in options.input_fnm): # hard coded demo
        print('-------------------Demo summary and CE-----------------------')
        options.input_rate = 0.05
        options.isdemo = 1
        options.col_name = ['ORDERKEY', 'PARTKEY', 'SUPPKEY', 'LINENUMBER', 'QUANTITY', 'EXTENDEDPRICE', 'DISCOUNT', 'TAX',
                    'RETURNFLAG', 'LINESTATUS', 'SHIPDATE', 'COMMITDATE', 'RECEIPTDATE', 'SHIPINSTRUCT', 'SHIPMODE']
        options.model_fnm = '../results/Irisv0.1-nml2Mnt2048nr128nm0.5.model'
        options.demoresult = '\tFor comparison, the following baseline results are pre-computed.\n\tSampling\tGMQ:3.02, 95th:109.98\n\txAVI\t\tGMQ:2.94, 95th:21.00\n\tLM-\t\tGMQ:2.21, 95th:8.08\n\tMSCN\t\tGMQ:3.62, 95th:52.0'
    return options
