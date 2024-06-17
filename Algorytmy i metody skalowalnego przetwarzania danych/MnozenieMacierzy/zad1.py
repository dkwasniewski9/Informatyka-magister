import sys

from pyspark import SparkContext

sc = SparkContext().getOrCreate()

vector = dict()


def map_vector(filename):
    with open(filename, 'r') as f:
        for line in f:
            temp = line.split(';')
            if len(temp) == 2:
                j, v_j = int(temp[0]), float(temp[1])
                vector[j] = v_j


def MapMatrixVector(line):
    ln = line.split(';')
    if len(ln) == 3:
        i, j, a_ij = int(ln[0]), int(ln[1]), float(ln[2])
        return [(i, a_ij * vector[j])]
    else:
        return []


def ReduceMatrixVector(line):
    i, values = line[0], list(line[1])
    return [(i, sum(values))]


if __name__ == "__main__":
    map_vector('v1.txt')
    matrix = sc.textFile('m1.txt')
    result = matrix.flatMap(MapMatrixVector).groupByKey().map(ReduceMatrixVector).collect()
    with open('zad1output.txt', 'w') as f:
        f.write(f"{result}\n")
