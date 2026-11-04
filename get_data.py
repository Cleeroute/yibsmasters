"""
get_data.py -- Download the real public datasets used in BUSINESS PROCESS ANALYTICS (2nd ed.).
Run once:  python get_data.py          (writes into ./real)

Every file is free for teaching use; cite the source in your reports (see Appendix B of the notes).
If a download fails (slow network, site moved), copy the file from the class shared drive / USB
into ./real with the same name and run the script again: it skips files that already exist.
"""
import io, os, sys, zipfile, urllib.request

REAL = "real"
os.makedirs(REAL, exist_ok=True)

SOURCES = {
    # Week 1, 3, 4, 6 -- UCI Online Retail II (Chen 2012), CC BY 4.0, doi:10.24432/C5CG6D
    "online_retail_II.xlsx": "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip",
    # Week 2 -- Receipt phase of an environmental permit process (Buijs, 4TU.ResearchData,
    # doi:10.4121/12709127.v2); copy distributed with the PM4Py test data
    "receipt.xes": "https://raw.githubusercontent.com/process-intelligence-solutions/pm4py/release/tests/input_data/receipt.xes",
    # Week 2 -- Numenta Anomaly Benchmark (Lavin & Ahmad 2015): industrial machine temperature
    "machine_temperature_system_failure.csv": "https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/machine_temperature_system_failure.csv",
    "combined_windows.json": "https://raw.githubusercontent.com/numenta/NAB/master/labels/combined_windows.json",
    # Week 7 -- Hillstrom MineThatData e-mail test (2008), 64,000 customers, randomised
    "hillstrom.csv": "http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv",
}


def fetch(name, url):
    dest = os.path.join(REAL, name)
    if os.path.exists(dest):
        print(f"[skip] {name} already present")
        return
    try:
        with urllib.request.urlopen(url, timeout=120) as r:
            data = r.read()
        if url.endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                member = [m for m in z.namelist() if m.endswith(name)][0]
                data = z.read(member)
        open(dest, "wb").write(data)
        print(f"[ok]   {name}  ({len(data) / 1e6:.1f} MB)")
    except Exception as e:                                   # keep going; report what to do
        print(f"[FAIL] {name}: {e}\n       -> copy it from the shared drive into ./{REAL}/")


for name, url in SOURCES.items():
    fetch(name, url)

# Convert the 45 MB Excel workbook to a 7 MB Parquet file once (Excel is slow to read).
xlsx, pq = os.path.join(REAL, "online_retail_II.xlsx"), os.path.join(REAL, "online_retail_II.parquet")
if os.path.exists(xlsx) and not os.path.exists(pq):
    import pandas as pd
    print("converting Online Retail II to Parquet (about one minute)...")
    sheets = pd.read_excel(xlsx, sheet_name=None, dtype={"Invoice": str, "StockCode": str, "Description": str})
    pd.concat(sheets.values(), ignore_index=True).to_parquet(pq, index=False)
    print(f"[ok]   online_retail_II.parquet")

# Week 7 -- organ-donation panel ships inside the causaldata package (pip install causaldata).
try:
    import causaldata  # noqa: F401
    print("[ok]   causaldata installed (organ_donations panel available)")
except ImportError:
    print("[FAIL] pip install causaldata")
sys.exit(0)
