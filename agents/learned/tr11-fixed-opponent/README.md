# PTCG Playground: public baseline agent, patched for the new cards

The first-edition agent of Team Unown Gradiant (Dipam Chakraborty), with a small
compatibility patch so it runs on the card pool of **The Pokémon Company - PTCG AI
Battle Challenge Playground**. Use it with the companion notebook, which adds the
competition's engine SDK and card sheet from the competition data and writes a
ready-to-submit `submission.tar.gz`.

## Contents

| path | what |
|---|---|
| `main.py` | Kaggle-format agent: greedy (argmax) policy, no search |
| `model.pt` | trained weights (token transformer, d256, 4 layers, 8 heads) |
| `deck.csv` | the original 60-card deck (Teal Mask Ogerpon ex / Hydrapple ex / Meganium) |
| `ptcg/` | the agent's Python package (observation encoder, tracker, model) |
| `data/` | static tables the agent reads (`embed_pca.npz`, `lingering_effects.json`) |
| `LICENSE` | MIT license of the original code and weights |

Not included on purpose: the competition engine (`cg/`) and the card sheet. Those come
from the competition data and are only for use in the competition.

## The patch

The playground adds 164 cards (ids 1268-1431) and 199 attacks (ids 1557-1755) that the
model never saw during training. The patch changes two files:

- `ptcg/rl/model.py`: new cards read the model's UNK card row and new attacks read
  the empty attack row, instead of indexing past the trained tables.
- `ptcg/rl/cards.py`: new ids are skipped when building the static feature tables,
  and the environment tables are sized to the engine's id space.

The agent's own deck uses only first-edition cards, so the agent plays it exactly as
the original did. The patch only matters when an opponent plays new cards.

## Source and license

Original code and weights: https://github.com/dipamc/kaggle-ptcg-ai-battle (MIT,
Copyright (c) 2026 Dipam Chakraborty). The compatibility patch is released under the
same MIT license.
