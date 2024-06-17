from collections import defaultdict

from pyspark import SparkContext


def Map(line):
    t = line.split()
    if len(t) == 4:
        if t[0] == 'A':
            i = int(t[1])
            j = int(t[2])
            aij = float(t[3])
            return [(j, ('A', i, aij))]
        if t[0] == 'B':
            j = int(t[1])
            k = int(t[2])
            bjk = float(t[3])
            return [(j, ('B', k, bjk))]
    return []


def Reduce(line):
    key, values = line[0], list(line[1])
    A = defaultdict(dict)
    B = defaultdict(dict)
    for value in values:
        if value[0] == 'A':
            i = value[1]
            Aij = value[2]
            A[i][key] = Aij
        elif value[0] == 'B':
            k = value[1]
            Bjk = value[2]
            B[k][key] = Bjk

    results = defaultdict(float)
    for i in A:
        for k in B:
            sum_product = sum(A[i][j] * B[k][j] for j in A[i] if j in B[k])
            key = (i, k)
            results[key] = sum_product
    return [(key, results[key]) for key in results]


if __name__ == "__main__":
    sc = SparkContext().getOrCreate()

    matrixes = sc.textFile("MM_basic.txt")

    result = matrixes.flatMap(Map).groupByKey().flatMap(Reduce).collect()

    with open('zad4output.txt', 'w') as f:
        f.write(f"{result}\n")
