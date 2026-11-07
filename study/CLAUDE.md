# study/ — rules

Study material on tQuant: Jupyter notebooks (`notebooks/`) + three LaTeX documents (`docs/`).
Everything is in **English**. The material grows during the guided review (see `ROADMAP.md`): each step adds a
notebook and updates the three documents together.

## docs/ — three documents, shared preamble

| Document | Main file | Chapters | Content |
|---|---|---|---|
| Exercises | `exercises.tex` | `chapters/equity.tex`, `chapters/rates.tex` | questions, answers, exercises |
| Models and Pricing | `models_and_pricing.tex` | `models/market_handles.tex`, `models/models.tex`, `models/products.tex` | definitions as implemented: market handles, simulation models, products (analytic and MC pricing); no teaching material (that goes in Exercises) |
| Architecture | `architecture.tex` | `architecture/*.tex` | technical, one chapter per block of the package-layout diagram (`architecture/layout_diagram.tex`): overview (layout, AAD), market data, market environment, instruments, pricers, models (+ calibration), analytics, foundations (dates, numerics), testing |

- `preamble.tex` is shared: packages, math macros (`\E`, `\Q`, `\dd`, ...), `\nb{...}` (notebook reference),
  `\code{...}` (library names), `\pending` (section still to be written). Macros live only there; the Q/A
  environments and tags stay in `exercises.tex`.
- One `\section` per asset class (Q/A), per chapter (Models and Pricing: Market handles, Models, Products) or per
  topic (Architecture); topics are `\subsection`s with a `\label` (`sec:mp-...`, `sec:arch-...`); Q/A and
  Architecture also with a `\nb{...}` to the notebook that shows them.
- Models and Pricing: purely functional / modelling, **no library references** (no class, method, parameter or
  key names: those are in Architecture). Per model: dynamics, calibration to the market step by step (what is
  fitted, to what, with which weights and bounds), closed forms, simulation scheme, sensitivities; key formulas,
  short, no long derivations. Architecture: the real classes, methods and files (check the
  code), how they connect; no financial derivations. Each fact in one document, cross-referenced from the others.
- References: `docs/refs.bib` (BibTeX, natbib author-year, `plainnat`; bibliography at the end of Models and
  Pricing). Cite the source of every model, formula and numerical method (`\citet` / `\citep`, chapter or equation
  when checked); add only references whose details are certain (books in `resources/`, standard papers).
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

- Name: `topic.ipynb` (`market_handles`, `dividends`, ...), no number prefix.
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
