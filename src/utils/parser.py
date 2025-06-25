import argparse


def get_args(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--cfg', '--config', dest='cfg_files', action='append',
        help='A nessessary config file.', default=None, type=str)
    parser.add_argument('--mode', help='Train or test mode.', default='test', 
        choices=['train', 'test', 'meta', 'mix'], type=str)
    args = parser.parse_args()
    
    if args.cfg_files is None:
        raise ValueError('No config file')
    
    return args