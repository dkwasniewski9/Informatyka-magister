import sys

lines = [line.strip() for line in sys.stdin]
sorted_lines = sorted(lines)

for line in sorted_lines:
    print(line)
