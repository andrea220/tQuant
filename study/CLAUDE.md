# study/ — rules

Study material on tQuant: Jupyter notebooks (`notebooks/`) + LaTeX Q/A (`docs/`).
Everything is in **English**.

## docs/ (LaTeX Q/A)

- One `\section` per asset class, one file each: `chapters/equity.tex`, `chapters/rates.tex`.
  Topics are `\subsection`s. Shared topics (numerics, AAD) go where first used, with a cross-reference.
- Always a `question` + `answer` pair, with one tag (`\Theory` or `\Practice`; interview/desk
  questions are `\Theory`) and a difficulty `\diff{1..3}`:
  ```latex
  \begin{question}{\Theory \diff{2}}{Short title}
    Question text. \nb{01\_market\_handles.ipynb}
  \end{question}
  \begin{answer}
    Answer.
  \end{answer}
  ```
- `\Practice` questions point to a notebook with `\nb{...}` and use the real tQuant API.
- Answers are short: key formula + intuition, no long derivations.
- Math macros live only in `main.tex` (`\E`, `\Q`, `\dd`, ...).
- `\showanswersfalse` in `main.tex` -> questions-only version.
- After every edit: `latexmk` in `docs/` (output in `docs/build/`), fix errors and warnings.

## notebooks/

- Name: `NN_topic.ipynb` (`01_market_handles`, `02_forwards`, ...).
- Straight to the point: one-line markdown per section, the code does the talking.
  Only the key formula where it helps; no prose recaps.
- Structure: title + one line on scope and related Q/A chapter -> imports -> market -> one
  section per concept (compute, print or plot) -> 1-3 short exercises.
- Use the synthetic market (`examples/market_data.build_market_environment`) so it runs anywhere.
- Real tQuant API only: check the code, do not invent methods. Reuse `examples/` logic, don't copy whole scripts.
- Small, self-contained cells; plots inline (no `savefig`, no `matplotlib.use("Agg")`).
- Must run top to bottom (Restart & Run All) in the env with `tensorflow` + `pip install -e .`.
