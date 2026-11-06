"""Download the six daily index series used in the paper from Yahoo Finance.

The raw price files are not redistributed in this repository (Yahoo Finance terms of use).
Run this script once; it writes data/<NAME>.csv and checks each file against data/SHA256SUMS.
Yahoo occasionally revises historical values; a checksum mismatch is reported but not fatal.
"""
import csv, datetime as dt, hashlib, json, os, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
TICKERS = {"GSPC": "^GSPC", "IXIC": "^IXIC", "DJI": "^DJI", "NSEI": "^NSEI", "BVSP": "^BVSP", "N225": "^N225"}
START, END = "2010-01-01", "2020-01-01"          # sample: 2010-01-04 .. 2019-12-31


def epoch(s):
    return int(dt.datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc).timestamp())


def fetch(symbol):
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(symbol)}"
           f"?period1={epoch(START)}&period2={epoch(END)}&interval=1d&events=history")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research-replication script)"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)["chart"]["result"][0]


def to_rows(res):
    off = res["meta"]["gmtoffset"]
    q = res["indicators"]["quote"][0]
    rows = []
    for i, ts in enumerate(res["timestamp"]):
        vals = [q[k][i] for k in ("open", "high", "low", "close")]
        if any(v is None for v in vals):
            continue
        rows.append([dt.datetime.fromtimestamp(ts + off, dt.timezone.utc).strftime("%Y-%m-%d"), *vals])
    return rows


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    import urllib.parse  # noqa: F401  (used in fetch)
    os.makedirs(DATA, exist_ok=True)
    sums = {}
    sf = os.path.join(DATA, "SHA256SUMS")
    if os.path.exists(sf):
        for line in open(sf):
            h, name = line.split()
            sums[name] = h
    ok = True
    for name, sym in TICKERS.items():
        out = os.path.join(DATA, name + ".csv")
        rows = to_rows(fetch(sym))
        with open(out, "w", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["date", "open", "high", "low", "close"])
            w.writerows(rows)
        match = sums.get(name + ".csv") == sha256(out)
        ok &= match
        print(f"{name:5s} {len(rows):5d} rows  {rows[0][0]} .. {rows[-1][0]}  checksum {'OK' if match else 'DIFFERS'}")
        time.sleep(1)
    if not ok:
        print("\nSome files differ from the versions used in the paper (Yahoo revises data). "
              "Results should be close but may not match to the last digit.", file=sys.stderr)


if __name__ == "__main__":
    import urllib.parse
    main()
