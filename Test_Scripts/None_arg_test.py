import argparse

parser = argparse.ArgumentParser()
parser.add_argument('s')
args = parser.parse_args()
print(type(args.s))
