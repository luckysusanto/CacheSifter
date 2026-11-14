from collections import namedtuple
from collections.abc import Generator
from sys import argv

import pandas as pd

Entry = namedtuple("Entry", ["timestamp", "vfs_kind", "path", "io_size", "file_size", "offset", "ino", "pid", "comm"])

def parse_lines(lines: list[str]) -> Generator[Entry, None, None]:
    for line in lines:
        if ',vfs_r,' in line or ',vfs_w,' in line:
            _, vals = line.rstrip('\n').split('->', 1)
            timestamp, vfs_kind, path, io_size, file_size, offset, ino, pid, comm = vals.split(',')
            timestamp, io_size, file_size, offset, ino, pid = map(int, (timestamp, io_size, file_size, offset, ino, pid))
            yield Entry(timestamp, vfs_kind, path, io_size, file_size, offset, ino, pid, comm)
        elif ',unlink,' in line:
            _, vals = line.rstrip('\n').split('->', 1)
            timestamp, _unlink, _f2fs, path, \
                _since_atime, _since_mtime, _since_ctime, \
                _file_size, _ino, _pid, _comm, _uid = vals.split(',')
            timestamp = int(timestamp)
            # Not handled.

def do_parse_lines(lines: list[str]) -> list[Entry]:
    return list(parse_lines(lines))

def filter_cache(entries: list[Entry]) -> list[Entry]:
    # Hopefully, this dumb filtering works
    return [e for e in entries if '/cache/' in e.path]

def group_by_path(entries: list[Entry]) -> dict[str, list[Entry]]:
    ans = {e.path: [] for e in entries}
    for e in entries:
        ans[e.path].append(e)
    return ans

def _lifetime(entries: list[Entry]) -> int:
    tss = {e.timestamp for e in entries}
    return max(tss) - min(tss)

def file_lifetimes(grouped_entries: dict[str, list[Entry]]) -> pd.DataFrame:
    """
    Given log entries grouped by filename, returns a dataframe with
    each file's lifetime (time difference between last access and first access)
    in seconds.
    """
    return pd.DataFrame(
        data=((path, _lifetime(entries)) for path, entries in grouped_entries.items()),
        columns=["path", "lifetime"],
    )

def _file_io_stats_per_sec(
    grouped_entries: dict[str, list[Entry]],
    num_sec: int = 20,
) -> Generator[tuple, None, None]:
    for path, entries in grouped_entries.items():
        rows = []
        read_size = read_count = write_size = write_count = file_size = last_access = 0
        first_access = now = min(e.timestamp for e in entries)
        for entry in entries:
            assert entry.timestamp >= now
            while entry.timestamp > now and len(rows) < num_sec:
                rows.append((
                    path,
                    now - first_access,
                    read_size,
                    read_count,
                    write_size,
                    write_count,
                    file_size,
                    last_access - first_access
                ))
                now += 1
                read_size = read_count = write_size = write_count = 0
            if len(rows) >= num_sec:
                break

            last_access = now
            if entry.vfs_kind == 'vfs_r':
                read_count += 1
                read_size += entry.io_size
                file_size = entry.file_size
            elif entry.vfs_kind == 'vfs_w':
                write_count += 1
                write_size += entry.io_size
                file_size = entry.file_size + entry.io_size
            else:
                raise ValueError(f"Unknown vfs_kind {entry.vfs_kind!r}")
        else:
            rows.append((
                path,
                now - first_access,
                read_size,
                read_count,
                write_size,
                write_count,
                file_size,
                last_access - first_access
            ))
        yield from rows

def file_io_stats_per_sec(grouped_entries: dict[str, Entry], num_sec: int = 20) -> pd.DataFrame:
    """
    Given log entries grouped by filename and the `num_sec`, the number of seconds of interest
    (default 20), return file stats for each file during the first `num_sec` seconds of it existence.

    Every row describes `k`th second of the file's lifetime: the rows are:
    - path - path to the file
    - k - relative timestamp (every number in `range(0, num_sec)`)
    - read_size - total size in bytes of reads by the end of `k`th second
    - read_count - total number of vfs_read calls by the end of `k`th second
    - write_size - total size in bytes of writes by the end of `k`th second
    - write_count - total number of vfs_write calls by the end of `k`th second
    - file_size - file size in bytes at the end of `k`th second
    - active_period - the difference between last and first time access by the end of `k`th second
    """
    return pd.DataFrame(
        data=_file_io_stats_per_sec(grouped_entries, num_sec),
        # k is the relative timestamp
        columns=["path", "k", "read_size", "read_count", "write_size", "write_count", "file_size", "active_period"],
    )

if __name__ == '__main__':
    with open(argv[1], "r") as f:
        entries = do_parse_lines(f.readlines())
        filtered = filter_cache(entries)
        grouped = group_by_path(filtered)

        lifetimes = file_lifetimes(grouped)
        print(lifetimes)
        lifetimes.describe()

        stats = file_io_stats_per_sec(grouped)
        print(stats)
        stats.describe()
