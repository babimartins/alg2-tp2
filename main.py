import numpy
import argparse
from kcenters import KCenters

parser = argparse.ArgumentParser()

# filename
parser.add_argument('-i', '--filename')
# constant for Minkowski distance
parser.add_argument('-p', '--p', type=int, default=2)
# index of class column
parser.add_argument('-c', '--c', type=int, default=0)

args = parser.parse_args()

S = numpy.genfromtxt(args.filename, delimiter=',')

kcenters = KCenters(S, args.c, args.p)
result = kcenters.calculateResults()

output = args.filename.split('.')[0] + ".out"
o = open(output, 'w')
o.write(result)
o.close