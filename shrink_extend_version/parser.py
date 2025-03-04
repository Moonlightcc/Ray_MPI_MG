import argparse


def get_parser():
    parser = argparse.ArgumentParser(
        description="Process MPI inputs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--gridsize', type=int, default=16,
        help='Number of point per axis')
    parser.add_argument('--initrange', type=int, default=1,
        help='Number of initial point per process along z aixs')
    parser.add_argument('--itn', type=int, default=4, 
        help='Number of iteration of multigrid algorithm')
    return parser
