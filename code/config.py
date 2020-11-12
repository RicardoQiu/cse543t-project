##################################################################
# This module is created to host all input configurations to run
# our model either training or inferring
##################################################################
import os
import argparse


def config():
    # add more configurations as we go, this is just a sample
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true', help='enables cuda')

    opt = parser.parse_args()
    print(opt)

    return opt



