# Blokus Duo AI

An **AlphaZero-style** AI for **Blokus Duo** (2-player, 14×14): a policy/value
neural network guided by Monte Carlo Tree Search and trained entirely by
self-play. Comes with a web GUI so you — and your friends — can play against it.

> **Status:** the full system is built and tested (engine, network, MCTS,
> self-play training, evaluation, and GUI). What remains is the long **training
> run on Colab** that climbs the strength ladder, and deploying the GUI. The code
> runs end-to-end today; the bundled model just isn't strong until trained.

## Quickstart

```bash
pip install -r requirements.txt
python -m pytest                 # 65 tests (engine fuzz, net, MCTS, arena, training, GUI)
python scripts/bench.py          # characterize the engine
python scripts/play_baselines.py # watch the heuristic bots play each other
python app.py                    # launch the GUI locally (weak until trained)
```

## How it works

- **One shared engine** (`blokus_core/`) is used by both training and the GUI, so
  the rules can never drift between them.
- **Self-play** generates games with MCTS; the network learns to predict the
  search's move distribution (policy) and who will win (value); a stronger
  network then makes stronger search — the AlphaZero loop.
- **Move generation** is the throughput bottleneck, so it has a Numba-JIT fast
  path (`blokus_core/movegen_nb.py`) validated against a brute-force reference.

## Repository layout

```
blokus_core/   shared engine
  rules · pieces · board · movegen · movegen_nb · encode · net · mcts
training/      config · selfplay · replay · train · loop  (checkpoint/resume)
eval/          baselines (random/greedy/blocking) · arena (paired games)
gui/           render · game_io · app   (Gradio)
notebooks/     train_colab.ipynb        (one-click Colab training)
scripts/       bench.py · train.py · play_baselines.py
tests/         65 tests incl. the move-gen fuzz test
app.py         GUI entry point (also used by Hugging Face Spaces)
```

## Train it (Colab)

Open `notebooks/train_colab.ipynb` in Colab (GPU runtime). It mounts Drive,
runs the test suite, enables the Numba fast path, and trains with **automatic
resume** — if the session drops, just re-run the training cell.

After pushing this repo to GitHub, add an Open-in-Colab badge:
`https://colab.research.google.com/github/<you>/<repo>/blob/main/notebooks/train_colab.ipynb`

Training saves checkpoints to Drive. Point the GUI at the latest one via the
`BLOKUS_WEIGHTS` environment variable (or copy it to `models/blokus_net.pt`).

## Play against it

- **Locally:** `python app.py` (add `SHARE=1` for a temporary public link).
- **Hugging Face Spaces** (free, permanent link to send friends): create a
  Gradio Space, push this repo, and add this front-matter to the Space's
  `README.md`:

  ```yaml
  ---
  title: Blokus Duo AI
  emoji: 🟦
  colorFrom: blue
  colorTo: red
  sdk: gradio
  app_file: app.py
  ---
  ```

  Upload your trained weights to the Space (or have it download them) and set
  `BLOKUS_WEIGHTS` accordingly.

## Roadmap

1. ✅ Engine: rules, pieces, board, move generation, encoding (fuzz-tested)
2. ✅ Network (PyTorch) + AlphaZero MCTS
3. ✅ Self-play training loop with Colab checkpoint/resume
4. ✅ Baselines + evaluation arena (color-swapped paired games)
5. ✅ Numba-accelerated move generation
6. ✅ Gradio GUI + Hugging Face Spaces entry point
7. ⏳ The long Colab training run → climb to Tier A
8. ⏳ (stretch) batched self-play inference; browser-only build on GitHub Pages

### Strength target (Tier A)

Measured with color-swapped paired games: ≥99% vs random, ≥90% vs the
blocking-greedy heuristic, ≥75% vs pure MCTS at 1600 simulations.

## License

MIT — see `LICENSE`.
