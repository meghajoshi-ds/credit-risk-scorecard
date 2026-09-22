"""Convert percent-format .py sources into .ipynb notebooks."""
import sys, pathlib, nbformat

def build(src, dst):
    text = pathlib.Path(src).read_text()
    cells, kind, buf = [], "code", []

    def flush():
        body = "".join(buf).strip("\n")
        if not body:
            return
        if kind == "markdown":
            body = "\n".join(
                line[2:] if line.startswith("# ") else ("" if line.strip() == "#" else line)
                for line in body.split("\n")
            )
            cells.append(nbformat.v4.new_markdown_cell(body))
        else:
            cells.append(nbformat.v4.new_code_cell(body))

    for line in text.splitlines(keepends=True):
        if line.startswith("# %%"):
            flush()
            buf = []
            kind = "markdown" if "[markdown]" in line else "code"
        else:
            buf.append(line)
    flush()

    nb = nbformat.v4.new_notebook(cells=cells)
    nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nbformat.write(nb, dst)
    print(f"{dst}: {len(cells)} cells")

if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2])
