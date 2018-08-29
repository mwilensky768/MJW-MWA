import argparse

parser = argparse.ArgumentParser()
parser.add_argument('s')
args = parser.parse_args()
if args.s is None:
    print('s is None')
elif args.s == None:
    print('s == None')
elif args.s is 'None':
    print('s is str(None)')
elif args.s == 'None':
    print('s == str(None)')
    args.s = None
    print(type(args.s))
