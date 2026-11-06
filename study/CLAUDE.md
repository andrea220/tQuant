# study/ — rules

Study material on tQuant: Jupyter notebooks (`notebooks/`) + three LaTeX documents (`docs/`).
Everything is in **English**. The material grows during the guided review (see `ROADMAP.md`): each step adds a
notebook and updates the three documents together.

## docs/ — three documents, shared preamble

| Document | Main file | Chapters | Content |
|---|---|---|---|
| Exercises | `exercises.tex` | `chapters/equity.tex`, `chapters/rates.tex` | questions, answers, exercises |
| Models and Pricing | `models_and_pricing.tex` | `models/equity.tex`, `models/rates.tex` | functional: dynamics, pricing formula / algorithm, calibration, greeks |
| Architecture | `architecture.tex` | `architecture/*.tex` | technical: market data (snapshots, keys, risk factors), class architecture, numerics, testing |

- `preamble.tex` is shared: packages, math macros (`\E`, `\Q`, `\dd`, ...), `\nb{...}` (notebook reference),
  `\code{...}` (library names), `\pending` (section still to be written). Macros live only there; the Q/A
  environments and tags stay in `exercises.tex`.
- One `\section` per asset class (Q/A, Models and Pricing) or per topic (Architecture); topics are `\subsection`s
  with a `\label` (`sec:mp-...`, `sec:arch-...`) and a `\nb{...}` to the notebook that shows them.
- Models and Pricing: what is computed, not how the code is organised; key formulas, the calibration and how
  the greeks are obtained; short, no long derivations. Architecture: the real classes, methods and files (check the
  code), how they connect; no financial derivations. Each fact in one document, cross-referenced from the others.
- Replace `\pending` when a subsection is written; keep the headings in the review order.
- After every edit: `latexmk` in `docs/` (builds the three documents into `docs/build/`), fix errors and warnings.

### Exercises (`exercises.tex`)

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
- Shared topics (numerics, AAD) go where first used, with a cross-reference.
- `\showanswersfalse` in `exercises.tex` -> questions-only version.

## notebooks/

- Name: `NN_topic.ipynb` (`01_market_handles`, `02_forwards`, ...).
- Straight to the point: one-line markdown per section, the code does the talking.
  Only the key formula where it helps; no prose recaps.
- Structure: title + one line on scope and related Q/A chapter -> imports -> market -> one
  section per concept (compute, print or plot) -> 1-3 short exercises.
- Use the synthetic market in `marketdata/` (`tq.MarketEnvironment.from_folder(REPO / "marketdata")`, latest
  snapshot; set `tq.Settings.evaluation_date = mkt.evaluation_date`) so it runs anywhere. Strikes and spot ranges
  as a fraction of spot, not hardcoded levels (snapshots change).
- Real tQuant API only: check the code, do not invent methods. Reuse `examples/` logic, don't copy whole scripts.
- Small, self-contained cells; plots inline (no `savefig`, no `matplotlib.use("Agg")`).
- Must run top to bottom (Restart & Run All) in the env with `tensorflow` + `pip install -e .`.
