# Why the Two-Tower Model Works

Notes on `module_4_3_recommender_colab.ipynb`'s `TwoTowerModel` — why the architecture is shaped the way it is, how an item's vector actually gets corrected during training, and why that vector can be reused across completely different users instead of being specific to whoever it was trained against.

## What it is

`TwoTowerModel` is MF (Matrix Factorization) with each side upgraded from a plain embedding lookup into a small MLP (Multi-Layer Perceptron):

```
MF:          user_emb[u]  ·  item_emb[i]
Two-Tower:   user_tower(user_emb[u])  ·  item_tower(item_emb[i], genre_features[i])
```

The item tower additionally fuses in side features (genre flags, in this notebook) before its MLP. Both towers still end in a single dense vector, combined with a plain dot product — same combiner as MF.

## Why the architecture works: independent towers, combined only at the end

The one property that matters most: **`user_vector()` and `item_vector()` never see each other's inputs.** Compare this to NCF (Neural Collaborative Filtering), which concatenates the user and item vectors *before* its MLP — you can't compute anything until you know both the specific user and the specific item.

Because the two towers stay independent until the final dot product:

- The **item tower can run once, offline, over the entire catalog** — producing a fixed table of item vectors, independent of any particular user. This table gets indexed into an ANN (Approximate Nearest Neighbour) structure (FAISS, ScaNN, HNSW in production; this notebook's `similar_movies_two_tower()` brute-force cosine scan is the tiny-scale stand-in).
- At request time, only the **user tower runs**, producing one query vector, which does a fast ANN lookup against that prebuilt item index — no need to re-score every item for every request.

NCF's entangled architecture can't do this — it has no way to precompute anything before seeing a specific `(user, item)` pair. That's why NCF-style models get used as a stage-2 *ranker* (scoring a few hundred already-retrieved candidates) rather than a stage-1 *retriever*.

## How an item vector actually gets nudged toward its correct value

The item tower's output for a given movie is a function of `item_emb[i]` and `genre_features[i]`, run through `Linear → ReLU → Linear`. During backpropagation, gradient flows through that whole chain, exactly like it flows through the user tower — the dot product `(u * v).sum()` treats both sides symmetrically:

```
∂logit/∂u = v     (gradient into the user vector)
∂logit/∂v = u     (gradient into the item vector — same rule, sides swapped)
```

A tiny concrete example, batch size 2, `n_factors=3` (toy — real run uses 32):

```
user0 = [0.8, -0.2, 0.5]      item0 = [0.6, 0.1, 0.9]   ← user0 actually liked this (label=1)
user2 = [-0.5, 0.2, 0.6]      item1 = [-0.3, 0.5, 0.2]  ← randomly sampled, unrated (label=0)
```

Forward pass (dot product of the two tower outputs — here treated as already-computed, to isolate the point):

```
logit0 = dot(user0, item0) = 0.8*0.6 + -0.2*0.1 + 0.5*0.9 = 0.91
logit1 = dot(user2, item1) = -0.5*-0.3 + 0.2*0.5 + 0.6*0.2 = 0.37
```

Backward pass — `BCEWithLogitsLoss (Binary Cross-Entropy with Logits)` gives `∂loss/∂logit = sigmoid(logit) - label`:

```
∂loss/∂logit0 = sigmoid(0.91) - 1 = 0.713 - 1 = -0.287   (too low vs. a true positive → push up)
∂loss/∂logit1 = sigmoid(0.37) - 0 = 0.591 - 0 =  0.591   (too high vs. a true negative → push down)
```

That gradient scales *both* branches:

```
grad_item0 = -0.287 * user0 = [-0.230,  0.057, -0.144]   → item0 nudged toward user0's direction (raise their score)
grad_item1 =  0.591 * user2 = [-0.296,  0.118,  0.355]   → item1 nudged away from user2's direction (lower their score)
```

Each of these gradients flows backward through the item tower's `Linear → ReLU → Linear`, updating both the tower's weights and the specific row of `item_emb` that was looked up (`item0`'s row, `item1`'s row) — no other movie's row is touched by this particular training step.

## Why the same item vector can be reused across completely different users

`item_emb` and every layer of `item_tower` are **one shared set of parameters** across the entire catalog and every user. A given movie's row in `item_emb` gets a gradient nudge *every time it appears in any training pair, for any user* — not just once, against whichever user it happened to be paired with in one batch.

Concretely: if `item0` also gets rated by `user5` (loved it) and randomly sampled as a negative for `user9` later in training, both of those steps contribute their own gradient to that exact same row:

```
(user0, item0, label=1): grad_item0 = -0.287 * user0's vector
(user5, item0, label=1): grad_item0 = -0.194 * user5's vector   (a different nudge, same row)
(user9, item0, label=0): grad_item0 =  0.402 * user9's vector   (pushes the other way, same row)
```

`item0`'s final vector, after every epoch, is the net accumulated result of *every* user who ever appeared paired with it — not a private relationship with any single user. That's the actual mechanism behind "collaborative" filtering: an item's vector is shaped collaboratively, by everyone who interacted with it.

This is exactly what makes the vector reusable and precomputable at serving time. `recommend()` computes a score for `(user0, some_movie_user0_never_rated)` — a pair that never appeared together during training — and it's only meaningful because `some_movie`'s vector was already shaped by feedback from other users, and `user0`'s vector was shaped by feedback from other movies. If an item vector only meant something relative to the one user it first trained against, none of this would generalize to a single new prediction.

## Production reality: what actually gets retrained, and how often

`TwoTowerModel` has six learnable weight tensors. Using this notebook's actual instantiation (`n_users=6040, n_items=3706, n_factors=32, hidden=64, n_genres=18`):

| Weight | Shape | Params |
|---|---|---|
| `user_emb.weight` | `[6040, 32]` | 193,280 |
| `item_emb.weight` | `[3706, 32]` | 118,592 |
| `user_tower[0]` (Linear 32→64) | weight `[64,32]` + bias `[64]` | 2,112 |
| `user_tower[2]` (Linear 64→32) | weight `[32,64]` + bias `[32]` | 2,080 |
| `item_tower[0]` (Linear 50→64) | weight `[64,50]` + bias `[64]` | 3,264 |
| `item_tower[2]` (Linear 64→32) | weight `[32,64]` + bias `[32]` | 2,080 |

Sum: `193,280 + 118,592 + 2,112 + 2,080 + 3,264 + 2,080 = 321,408` — matches the `Two-Tower parameters: 321,408` the notebook prints when it builds the model. `item_tower[0]`'s input is `50 = 32 (item_emb) + 18 (genre flags)`; both towers' final `Linear` layers project back down to `32`, so the vectors actually compared via the dot product are `32`-dimensional on both sides, same size as plain MF's embeddings. Not counted here: `item_genres` (the `[3706, 18]` genre matrix) is a `register_buffer`, not a learnable parameter — it never receives gradients.

The key question for running this in production (a YouTube-style system, say) is: which of these six need to change per-request, and which can sit still between full retrains?

**The clean half — `item_emb.weight`, `item_tower[0]`, `item_tower[2]`:** these get updated on whatever the periodic retraining cadence is (a batch job, commonly somewhere in the daily-to-weekly range for large-scale systems — treat any more precise number as unverified), and are completely static in between. A video's final vector, once computed after a retrain, gets reused for every request from every user until the next retrain — that's exactly the property that makes offline precomputation + ANN (Approximate Nearest Neighbour) indexing possible.

**Where our toy model doesn't map cleanly onto a real system — `user_emb.weight`:** in this notebook, `user_emb.weight` is a static per-`user_id` lookup row, mechanically identical to `item_emb.weight`. A real production system typically has **no equivalent of this table at all.** Instead of a trained-per-ID row, the user tower's *input* is raw recent behavior (recent watch history, search tokens, etc.), computed fresh from a live behavioral stream. There's nothing here that "changes on each request" in the sense of a weight being updated — there's just new *input data* every request, flowing through weights that haven't moved since the last retrain. Our `user_emb.weight` is really standing in for "recent behavior as a feature," collapsed into a static per-user-ID lookup because this notebook only has static historical ratings to train on, not a live stream of behavior.

**`user_tower[0]` and `user_tower[2]`:** these behave exactly like the item tower's weights — periodically retrained (typically both towers together, end-to-end, in the same batch job), static between retrains, never touched per-request. What happens on every request is a cheap **forward pass** through these already-trained, currently-fixed weights, given that request's fresh behavioral input — inference, not training.

```
Retrained periodically (batch job, all together):
  item_emb.weight, item_tower[0], item_tower[2], user_tower[0], user_tower[2]
  ← none of these update per-request

Computed fresh every request (no training involved, just a forward pass):
  the user's query vector = user_tower(recent_watch_history)
  ← only the INPUT differs each time; the weights doing the computing are fixed until the next retrain
```
