import optparse
import os

def parse_arg():
    parser = optparse.OptionParser()

    parser.add_option('--cords_fnm', dest='cords_fnm', default='CORDS.log', type='str')

    parser.add_option('--nusecpp', dest="nusecpp", help="Use C++ code for speedup. Default: 1", default=1, type='int')
    parser.add_option('--nn', dest="nn", help="# FC layers. Default: 0.", default=0, type='int')
    parser.add_option('--nfc', dest="nfc", help="FC layers size. Default: 128.", default=128, type='int')

    parser.add_option('--storage', dest="storage", default=60, type='int')
    parser.add_option('--atom', dest="atom", default=512, type='int')

    parser.add_option('--nt', dest="nt", help="Total cell embedding budget. Default: 1024.", default=1024, type='int')
    parser.add_option('--nr', dest="nr", help="Summary size. Default: 128.", default=128, type='int')
    parser.add_option('--ne', dest="ne", help="Cell embedding length. Default: 128.", default=128, type='int')

    parser.add_option('--neb', dest="neb", help="Max column resolution: 128.", default=128, type='int')

    parser.add_option('--nlen', dest="nlen", help="Max training length. Default: 300000.", default=300000, type='int')
    parser.add_option('--normlen', dest="normlen", help="Norm length. Default: 1000.", default=1000, type='int')

    parser.add_option('--mind', dest='mind', help='Minimum sketch dimensions. Defalut: 2', default=2, type='int')
    parser.add_option('--maxd', dest='maxd', help='Maximum sketch dimensions. Defalut: 2', default=2, type='int')
    parser.add_option('--nb', dest="nb", help="Base size. Default: 2.", default=2, type='int')

    options, args = parser.parse_args()
    return options
