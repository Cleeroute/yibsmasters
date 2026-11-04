# Business Process Analytics (YIBS, MTech 2) - 2nd edition starter kit

- `requirements.txt` - pinned versions used to test every lab (Nov 2026)
- `datagen.py` - writes the Mboa Fresh Market teaching case to `./data`
- `get_data.py` - downloads the real public datasets to `./real` (copy from the shared drive if a site is unreachable)
- `notebooks/lab1..lab7.ipynb` - the seven guided labs, ready for Colab or JupyterLab
- `labs/lab*.py` - the same labs as plain scripts with `# %%` step markers
- `runlab.py` - runs a lab script step by step and saves each step's code and output to `out/` (used to build the notes)
- `latex/` - LaTeX source of the lecture notes; compile `latex/main.tex` with `latexmk -pdf` from inside `latex/`
  (it reads `../out` and `../figs`)

Quick start:
    pip install -r requirements.txt
    python datagen.py
    python get_data.py
    jupyter lab notebooks/
# yibsmasters
# yibsmasters
# yibsmasters
# yibsmasters
# yibsmasters
# yibsmasters
