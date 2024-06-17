import sys

from zad3mapper import mapper
from zad3reducer import reducer
from zad3sort import sort


def read_input(filename):
    with open(filename, 'r') as f:
        return f.readlines()


if __name__ == "__main__":
    input_filename = sys.argv[1]

    print('czytanie')
    input_data = read_input(input_filename)

    print('map')
    mapped_data = mapper(input_data)

    print('sort')
    sorted_mapped_data = sort(mapped_data)

    print('reduce')
    reduced_output = reducer(sorted_mapped_data)

    with open('output.txt', 'w') as f:
        for item in reduced_output:
            f.write(f"{item}\n")
