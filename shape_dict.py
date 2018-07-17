import argparse

parser = argparse.ArgumentParser()
parser.add_argument("outdir", nargs=1, action='store', help="Destination for pickled dictionary")
parser.add_argument('--keys', nargs=*, help="The names of the shapes - these can be whatever you want")
parser.add_argument('--lows', nargs=*, action='store', type=float, help="The lower bound of the shapes referred to by the keys")
parser.add_argument('--highs', nargs=*, action='store', type=float, help="The upper bounds of the shapes referred to by the keys")
