from collections import namedtuple
from collections.abc import Generator
from sys import argv

Entry = namedtuple("Entry", ["timestamp", "vfs_kind", "filename", "a", "b", "c", "d", "e", "f"])

def parse_lines(lines: list[str]) -> Generator[Entry, None, None]:
    for line in lines:
        if ',vfs_r,' in line or ',vfs_w,' in line:
            _, vals = line.rstrip('\n').split('->', 1)
            timestamp, vfs_kind, filename, a, b, c, d, e, f = vals.split(',')
            timestamp = float(timestamp)
            a, b, c, d, e = map(int, (a, b, c, d, e))
            yield Entry(timestamp, vfs_kind, filename, a, b, c, d, e, f)

def do_parse_lines(lines: list[str]) -> list[Entry]:
    return list(parse_lines(lines))

if __name__ == '__main__':
    with open(argv[1], "r") as f:
        parsed = do_parse_lines(f.readlines())
        print(f"Parsed {len(parsed)} entries, e.g.:")
        print(parsed[0])
        print(parsed[-1])
