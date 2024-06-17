from math import floor

from pyspark import SparkContext


def MapMatrixVector(line):
    ln = line.split()
    if len(ln) == 3:
        i, j, a_ij = int(ln[0]), int(ln[1]), float(ln[2])
        k = floor(j / slice_length)
        with open(f'v_{k}.txt', 'r') as f:
            for lineV in f:
                temp = lineV.split()
                if len(temp) == 2:
                    j, v_j = int(temp[0]), float(temp[1])
                    vector = v_j
        return [(i, a_ij * vector)]
    else:
        return []


def ReduceMatrixVector(line):
    i, values = line[0], list(line[1])
    return [(i, sum(values))]


sc = SparkContext().getOrCreate()
slice_length = 3

N = 9

vectors = [open(f'v_{k}.txt', 'w') for k in range(int(N / slice_length))]

matrices = [open(f'm_{k}.txt', 'w') for k in range(int(N / slice_length))]

input_file = open('joined.txt', 'r')
for line in input_file:
    l = line.split()
    if len(l) == 2:
        i, v = int(l[0]), float(l[1])
        k = int(i / slice_length)
        vectors[k].write(line)
    if len(l) == 3:
        i, j, aij = int(l[0]), int(l[1]), float(l[2])
        k = int(j / slice_length)
        matrices[k].write(line)
vector = dict()
k_old = -1
for file in vectors + matrices:
    file.close()

joined = sc.textFile('m_*.txt')
result = joined.flatMap(MapMatrixVector).groupByKey().map(ReduceMatrixVector).collect()

with open('zad2output.txt', 'w') as f:
    f.write(f"{result}\n")
