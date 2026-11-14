from collections import namedtuple
from collections.abc import Generator
from sys import argv

Entry = namedtuple("Entry", ["timestamp", "vfs_kind", "filename", "io_size", "file_size", "offset", "ino", "pid", "comm"])

def parse_lines(lines: list[str]) -> Generator[Entry, None, None]:
    for line in lines:
        if ',vfs_r,' in line or ',vfs_w,' in line:
            _, vals = line.rstrip('\n').split('->', 1)
            timestamp, vfs_kind, filename, io_size, file_size, offset, ino, pid, comm = vals.split(',')
            timestamp = float(timestamp)
            io_size, file_size, offset, ino, pid = map(int, (io_size, file_size, offset, ino, pid))
            yield Entry(timestamp, vfs_kind, filename, io_size, file_size, offset, ino, pid, comm)
        elif ',unlink,' in line:
            _, vals = line.rstrip('\n').split('->', 1)
            timestamp, _unlink, _f2fs, filename, \
                _since_atime, _since_mtime, _since_ctime, \
                _file_size, _ino, _pid, _comm, _uid = vals.split(',')
            timestamp = float(timestamp)
            # Not handled.

def do_parse_lines(lines: list[str]) -> list[Entry]:
    return list(parse_lines(lines))

if __name__ == '__main__':
    with open(argv[1], "r") as f:
        parsed = do_parse_lines(f.readlines())
        print(f"Parsed {len(parsed)} entries, e.g.:")
        print(*parsed[:5], sep='\n')
        print()
        print(parsed[-1])
