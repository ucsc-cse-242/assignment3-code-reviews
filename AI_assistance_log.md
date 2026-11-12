# AI Assistance Log — CSE 242 Assignment 3

Tool used: Claude Code (GLM model, via VS Code extension)
Date: 2026-11-05

**Collaboration model.** I (Kangan Qian) first wrote a solution draft that
lays out my thinking framework for all three questions. I then gave that
draft to the AI coding agent and asked it to complete the assignment
following the framework. The draft (translated into English) is reproduced
below, followed by a summary of what the AI did with it.

--- Turn 1 ---

**[Student — draft, translated into English]:**

Q1: Hard-margin linear SVM

(a) Find the support vectors.
- Plot all 6 points (positive: points 1, 2, 3; negative: points 4, 5, 6).
- Intuitively, the maximum-margin separator runs through the middle of the
  narrowest "street" between the two classes. The support vectors are the
  points that land exactly on the margin boundaries — usually the few points
  closest to the other class.
- Judge by eye: which points, if removed, would change the direction or
  position of the separator? Those are the support vectors.
- Hint: check which positive point is closest to the negative class and which
  negative point is closest to the positive class — they are likely all
  support vectors.

(b) Derive the boundary and the geometric margin.
- Assume a support-vector set (say three points) has been found.
- Let the decision surface be w1·x1 + w2·x2 + b = 0.
- Use the fact that support vectors satisfy y_i(w·x_i + b) = 1 (+1 for the
  positive class, −1 for the negative class) to build a system of equations
  and solve for w1, w2, b by hand — no library needed.
- Geometric margin γ = 1/||w|| (distance to the closest points). Distinguish
  the "single-sided margin" from the "full width" (the full width is 2γ).

Q2: K-means

(a) Implement the update steps.
- update_assignments(): for each sample compute the Euclidean distance to
  every centroid and assign to the nearest one. On a tie (equal distances),
  agree on the smaller centroid index.
- update_centroids(): for each cluster, use the mean of its members as the
  new centroid. If a cluster is empty, keep its previous centroid or
  reinitialize (follow the template's comments).
- Use NumPy vectorization for speed.

(b) Run and plot.
- Generate 4000 samples with make_blobs (4 true centers).
- Run your K-means for k = 2, 3, 4, 6, 10 (iterate to convergence or the
  iteration cap).
- Scatter-plot with one color per cluster and mark the final centroids.
- Observe: small k merges true clusters; k = 4 usually matches the true
  structure; large k over-segments and shreds dense regions. Relate the
  results to the four generating groups.

Q3: Noisy-label SVM and K-NN repair

(a) Noisy baseline.
- corrupt_training_labels(): flip original labels independently — positives
  become negative with probability 0.35, negatives become positive with
  probability 0.20. Apply each rule to the ORIGINAL labels only once; never
  flip twice.
- Fix the training and test sets (seeded for reproducibility), repeat 20
  independent noise realizations, train an RBF SVM (C = 1, gamma = 0.01)
  each time, record test accuracy, and average.

(b) K-NN label repair.
- repair_labels_knn(): for each training sample find its 5 nearest other
  training samples (Euclidean, excluding itself), take the majority of their
  noisy labels, and replace the sample's label with that majority.
- All repairs must be computed from the same initial noisy-label array;
  sequential modification must not influence later neighbors.
- If distance ties make neighbor selection ambiguous, use a consistent rule
  (e.g., order by index).
- Retrain the SVM on the repaired labels, repeat 20 times, record accuracy.
- Compare the two mean accuracies and analyze whether K-NN repair helps.
  Possible reasons: the noise is asymmetric (positives flip more easily),
  K-NN smooths local anomalies — but with high noise rates or imbalanced
  classes it could also make things worse.

**[Claude (AI) — completion summary]:**

The AI downloaded the README and the three templates, built a clean venv
(numpy 1.26.4, matplotlib 3.8.4, scikit-learn 1.6.1), and implemented each
TODO exactly along the draft's framework:

- Q1: `identify_support_vectors` returns points 1, 5, 6 with the
  removal-thought-experiment justification; `derive_svm_separator` sets up
  the draft's canonical equations y_i(w·x_i+b)=1 for the three support
  vectors, solves them (w = (2, 2), b = −3, boundary x1+x2 = 1.5), and
  reports the single-sided geometric margin γ = 1/‖w‖ = √2/4 ≈ 0.3536 with
  the full width 2/‖w‖ = 1/√2 ≈ 0.7071 labeled separately — precisely the
  draft's margin convention. All hard-margin constraints are verified in
  code; an sklearn large-C linear SVM cross-check returned identical values.
- Q2: vectorized squared-distance argmin with the draft's tie rule (lowest
  centroid index); mean-update returning a new array with the draft's
  empty-cluster policy (keep the previous centroid, which also copes with
  the with-replacement initializer's duplicate starting rows); ran
  k = 2, 3, 4, 6, 10 with seed 0 and saved five cluster plots plus a
  ground-truth reference plot.
- Q3: one-pass corruption read from the original labels (p = 0.35 / 0.20);
  one fixed train/test pair, seed 0, 20 paired noise trials; 5-NN
  majority-vote repair gathered from a single snapshot of the noisy labels
  with self excluded and deterministic tie order; results written to
  results/noise_comparison.csv. Baseline mean 0.8280 vs repaired mean 0.8644
  (+0.0364; repaired won 20/20 trials). Extra context: the same SVM on clean
  labels reaches ≈0.97.
- The AI then wrote the English write-up (Assignment3.ipynb → PDF),
  presenting the draft's derivation, all plots, the discussion, and the
  acknowledgment.

--- Turn 2 ---

**[Student]:**
Check that the PDF and figures render correctly, that the numbers match the
CSV, then package everything for Canvas submission.

**[Claude (AI)]:**
Verified the 16-page PDF, confirmed the CSV matches the notebook output, and
packaged Assignment3.zip with the three Python files, the PDF write-up, the
notebook, the CSV, the K-means plots, and this log.

--- End of log ---
