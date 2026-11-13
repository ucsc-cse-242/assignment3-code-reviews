ASSIGNMENT 3 SUBMISSION

Read Assignment3.pdf for the SVM derivation, K-means plots/discussion,
and noisy-label SVM comparison. The three completed source files are:
  assignment3_template.py
  kmeans_template.py
  noisy_classification_template.py

The experiment CSV is results/noise_comparison.csv (20 paired trials).
Individual plots and the K-means summary are under results/.
All AI disclosure and deidentified conversation/tool evidence is contained
in ai-disclosure-and-transcripts.zip. Extract it to read the disclosure PDF,
readable conversation, raw source JSONL, and scope/privacy manifest.

SETUP
Tested with Python 3.13.12; dependencies are pinned in requirements.txt.
From this folder, create a fresh environment:
  python3 -m venv .venv
  source .venv/bin/activate
  python -m pip install -r requirements.txt

RUN THE REQUIRED SOLUTIONS
  python assignment3_template.py --task plot
  python assignment3_template.py --task svm
  python kmeans_template.py --seed 0 --max-iterations 100
  python noisy_classification_template.py --repeats 20 --seed 0

Optional focused verification:
  python verify_solutions.py

REPRODUCIBILITY
K-means uses the specified 4,000-row make_blobs dataset and one NumPy
Generator seeded with 0, sampling centroid rows with replacement in the
order k=2,3,4,6,10. It uses 100 provided-loop iterations and final assignments.
Ties select the smallest centroid index; empty clusters keep their centers.

The noise experiment uses default_rng(0), generates training then independent
test features once, then draws fresh label corruption for each of 20 trials.
Original positive labels flip with probability .35; original negatives with
.20. The baseline and repaired method share exactly the same noisy labels
and fixed features per trial. Five-neighbor repair excludes self by row index,
breaks distance ties by the lower original row index, and votes from the
unchanged noisy array. It never receives clean training or test labels.
Every final classifier is a fresh RBF SVC(C=1, gamma=.01).

Mean baseline accuracy: 0.827950 (82.7950%).
Mean K-NN repaired accuracy: 0.864375 (86.4375%).
Difference: +0.036425 (+3.6425 percentage points).

UPLOAD
The folder contains normal assignment files plus a separate AI-materials ZIP.
Assignment3-Submission.zip is an optional archive of the whole folder.
Nothing has been submitted automatically.
