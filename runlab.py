"""Run a lab script chunk by chunk ('# %% name' markers), saving each chunk's code and output
so that the LaTeX notes include exactly the code that was executed and what it printed."""
import sys, io, re, contextlib, os, time
import matplotlib
matplotlib.use("Agg")

path = sys.argv[1]
lab = os.path.splitext(os.path.basename(path))[0]
src = open(path).read()
parts = re.split(r"^# %% (\S+)\s*$", src, flags=re.M)
ns = {"__name__": "__main__"}
os.makedirs("out", exist_ok=True)
preamble = parts[0]
if preamble.strip():
    exec(compile(preamble, f"{lab}-preamble", "exec"), ns)
for name, code in zip(parts[1::2], parts[2::2]):
    code = code.strip("\n") + "\n"
    buf = io.StringIO()
    t = time.time()
    with contextlib.redirect_stdout(buf):
        exec(compile(code, f"{lab}-{name}", "exec"), ns)
    out = buf.getvalue()
    open(f"out/{lab}_{name}.py", "w").write(code)
    open(f"out/{lab}_{name}.txt", "w").write(out)
    print(f"===== {lab}_{name}  ({time.time()-t:.1f}s)\n{out}")
