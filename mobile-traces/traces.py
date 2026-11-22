from collections import namedtuple
from collections.abc import Generator
from sys import argv

import pandas as pd

Entry = namedtuple("Entry", ["timestamp", "vfs_kind", "path", "io_size", "file_size", "offset", "ino", "pid", "comm"])

# from kernel-code/fs/f2fs/namei.c, is_cache_file()
CACHE_DIRECTORIES = (
    "/data/com.king.candycrushsaga/cache/",
    "/data/com.android.chrome/cache/",
    "/data/com.google.earth/cache/",
    "/data/com.facebook.katana/cache/",
    "/data/org.mozilla.firefox/cache/",
    "/data/com.google.android.apps.maps/cache/",
    "/data/com.twitter.android/cache/",
    "/data/com.google.android.youtube/cache/",
    "/data/com.ea.game.pvzfree_row/cache/",
    "/data/com.ss.android.ugc.aweme/cache/",
    "/data/com.ss.android.ugc.trill/cache/",
    "/data/cache/",
)

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
    # keep only those entries whose paths match cacheSifter's cache dir list
    return [e for e in entries if any(e.path.startswith(directory) for directory in CACHE_DIRECTORIES)]

def group_by_inode(entries: list[Entry]) -> dict[int, list[Entry]]:
    ans = {e.ino: [] for e in entries}
    for e in entries:
        ans[e.ino].append(e)
    return ans

def _lifetime(entries: list[Entry]) -> int:
    tss = {e.timestamp for e in entries}
    return max(tss) - min(tss)

def file_lifetimes(grouped_entries: dict[int, list[Entry]]) -> pd.DataFrame:
    """
    Given log entries grouped by filename, returns a dataframe with
    each file's lifetime (time difference between last access and first access)
    in seconds.
    """
    return pd.DataFrame(
        data=((ino, entries[0].path, _lifetime(entries)) for ino, entries in grouped_entries.items()),
        columns=["inode", "first_observed_path", "lifetime"],
    )

def _file_io_stats_per_sec(
    grouped_entries: dict[int, list[Entry]],
    num_sec: int = 20,
) -> Generator[tuple, None, None]:
    for ino, entries in grouped_entries.items():
        rows = []
        read_size = read_count = write_size = write_count = file_size = last_access = 0
        
        first_access_ms = min(e.timestamp for e in entries)
        previous_timestamp = first_access_ms
        now = 0

        for entry in entries:
            assert entry.timestamp >= previous_timestamp
            previous_timestamp = entry.timestamp

            # convert each entry's timestamp into whole seconds since the file's first access
            elapsed_seconds = (entry.timestamp - first_access_ms) // 1000

            while elapsed_seconds > now and len(rows) < num_sec:
                rows.append((
                    ino,
                    entry.path,
                    now,
                    read_size,
                    read_count,
                    write_size,
                    write_count,
                    file_size,
                    last_access
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
                # entry.file_size + entry.io_size won't work when a write overwrites existing data
                # instead of appending data to the end of the file
                file_size = max(entry.file_size, entry.offset + entry.io_size)
            else:
                raise ValueError(f"Unknown vfs_kind {entry.vfs_kind!r}")
        else:
            while len(rows) < num_sec:
                rows.append((
                    ino,
                    entry.path,
                    now,
                    read_size,
                    read_count,
                    write_size,
                    write_count,
                    file_size,
                    last_access
                ))
                now += 1
                read_size = read_count = write_size = write_count = 0
        yield from rows

def file_io_stats_per_sec(grouped_entries: dict[int, Entry], num_sec: int = 20) -> pd.DataFrame:
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
        columns=["inode", "path", "k", "read_size", "read_count", "write_size", "write_count", "file_size", "active_period"],
    )

if __name__ == '__main__':
    if len(argv) > 2:
        lifetimes_csv = argv[2]
        stats_csv = argv[3]
    else:
        lifetimes_csv = stats_csv = None

    with open(argv[1], "r") as f:
        entries = do_parse_lines(f.readlines())
        filtered = filter_cache(entries)
        grouped = group_by_inode(filtered)

        lifetimes = file_lifetimes(grouped)
        print(lifetimes.describe())
        if lifetimes_csv:
            lifetimes.to_csv(lifetimes_csv, index=False)

        stats = file_io_stats_per_sec(grouped)
        print(stats.describe())
        if stats_csv:
            stats.to_csv(stats_csv, index=False)
