from pyspark import SparkContext


def MapMM1(line):
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


def ReduceMM1(line):
    key, values = line[0], list(line[1])
    A = []
    B = []
    for item in values:
        if item[0] == 'A':
            A.append([item[1], item[2]])
        if item[0] == 'B':
            B.append([item[1], item[2]])
    results = []
    for a in A:
        for b in B:
            results.append(((a[0], b[0]), a[1] * b[1]))
    return results


def MapMM2(line):
    return line


def ReduceMM2(line):
    key, values = line[0], list(line[1])
    return [(key, sum(values))]


if __name__ == "__main__":
    sc = SparkContext().getOrCreate()

    matrixes = sc.textFile("MM_basic.txt")

    result = (matrixes.flatMap(MapMM1).groupByKey().map(ReduceMM1).flatMap(MapMM2).groupByKey().flatMap(ReduceMM2)
              .collect())

    with open('zad3output.txt', 'w') as f:
        f.write(f"{result}\n")
