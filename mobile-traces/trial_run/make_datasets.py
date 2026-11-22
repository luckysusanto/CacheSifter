import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N_FILES = 100

# Following paper's criteria
BAR_MAX, LONG_MIN = 30, 90
K1, K2 = 20, 60

# per second rows + active period per file
def make_file(i):
    # lifetime active period = last access - first access (seconds)
    r = RNG.random() # Decide file class
    if r < 0.40:
        ap = int(RNG.integers(0, BAR_MAX))               # randomize BAR lifetime
    elif r < 0.65:
        ap = int(RNG.integers(BAR_MAX, LONG_MIN + 1))    # randomize transient lifetime
    else:
        ap = int(RNG.integers(LONG_MIN + 1, 400))        # randomize long-living lifetime
 
    file_size = int(RNG.lognormal(4, 1.2)) + 1           # Meh, just some random numbers, assume each file is around 55 KB
    first, last = 1, 1 + ap
 
    rate = RNG.uniform(0.05, 0.5) # Rate of activity per second
    n_extra = RNG.poisson(rate * (ap + 1)) # Expected number of activity + some noise
    extra = first + np.floor((ap + 1) * RNG.beta(0.7, 1.5, size=n_extra)).astype(int) # Scatters accesses across lifetime
    times = np.sort(np.r_[first, extra, last if ap > 0 else []]).astype(int)
 
    n_rows = max(last, K2)
    ra, rc, wa, wc = (np.zeros(n_rows + 1) for _ in range(4))   # index = second k
    for j, t in enumerate(times):
        if j == 0 or RNG.random() < 0.2:                 # 20% write, 80% read, first access is always write
            wa[t] += file_size if j == 0 else RNG.integers(1, file_size + 1)
            wc[t] += 1
        else:
            ra[t] += RNG.integers(1, file_size + 1)  # Number of KB read or write
            rc[t] += 1
 
    k = np.arange(1, n_rows + 1)
    accessed = (ra + rc + wa + wc)[1:] > 0
    last_seen = np.maximum.accumulate(np.where(accessed, k, first))
 
    df = pd.DataFrame({
        "id": f"file_{i:03d}",
        "k": k,
        "read_amount": np.cumsum(ra[1:]).astype(int),
        "read_count": np.cumsum(rc[1:]).astype(int),
        "write_amount": np.cumsum(wa[1:]).astype(int),
        "write_count": np.cumsum(wc[1:]).astype(int),
        "file_size": file_size,
        "active_period": last_seen - first,              # so far, up to second k
    })
    df.insert(2, "io_sum", df.read_amount + df.write_amount)
    return df, ap

rows, lifetimes = zip(*(make_file(i) for i in range(N_FILES)))
raw = pd.concat(rows, ignore_index=True)
raw.to_csv("raw_dummy.csv", index=False)

# Generate Label
ids = [
    f"file_{i:03d}" for i in range(N_FILES)
]
ap = np.array(lifetimes)
split = np.array(["train"] * N_FILES, dtype=object)
perm = RNG.permutation(N_FILES)
split[perm[int(0.7 * N_FILES) : int(0.85 * N_FILES)]] = "val"
split[perm[int(0.85 * N_FILES):]] = "test"

labels = pd.DataFrame({
    'id': ids,
    'lifetime_active_period': ap,
    'is_bar': (ap < BAR_MAX).astype(int),
    'is_transient': ((ap >= BAR_MAX) & (ap <= LONG_MIN)).astype(int),
    'is_long': (ap > LONG_MIN).astype(int),
    'split': split
})
labels.to_csv('labels.csv', index=False)