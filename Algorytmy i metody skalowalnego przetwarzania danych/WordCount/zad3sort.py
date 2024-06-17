
def sort(data):
    lines = [line.strip() for line in data]
    sorted_lines = sorted(lines)
    result = []
    for line in sorted_lines:
        result.append(line)
    return result
