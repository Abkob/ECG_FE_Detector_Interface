# Five-beat RR/HRV-adjacent QRS morphology branch

## Implemented boundary

The morphology module is a deterministic waveform-change feature extractor.
It is not a beat classifier, artifact classifier, or seizure classifier.

```text
single-lead ECG + explicitly named timestamp track
  -> symmetric 120-ms waveform around every timestamp
  -> five consecutive waveforms Q_i (5 x m)
  -> uncentered Gram matrix S_i = Q_i Q_i^T
  -> five ordered eigenvalues lambda1 ... lambda5
  -> provenance, reference-label, and reliability context
```

The implementation accepts arbitrary timestamps instead of embedding one
detector. The same code can therefore be run with expert annotations, UNSW,
NeuroKit, Zhai, or a future patient-validated timestamp source. This prevents
detector behavior from being confused with the morphology calculation.

Primary reference: Varon et al. (2015), *Detection of epileptic seizures by
means of morphological changes in the ECG*,
<https://doi.org/10.1016/j.jelectrocard.2015.08.020>. The earlier 80-ms version
is traceable to Varon et al. (2013), <https://cinc.org/archives/2013/pdf/0863.pdf>.

## Published 2015 core

For a timestamp `r_j`, the branch captures 60 ms before and 60 ms after it.
The exact stored sample count is

\[
m=\operatorname{round}(0.120 f_s).
\]

At 250 Hz this is 30 samples; at 360 Hz it is 43 samples after rounding. The
anchor is included once. If an even-length capture prevents perfect sample
symmetry, the unavoidable one-sample difference is placed after the anchor
and is recorded in metadata.

Five consecutive waveforms are stacked:

\[
Q_i=
\begin{bmatrix}
q_i\\q_{i+1}\\q_{i+2}\\q_{i+3}\\q_{i+4}
\end{bmatrix}\in\mathbb{R}^{5\times m}.
\]

The branch then calculates

\[
S_i=Q_iQ_i^T
\]

and exports its eigenvalues in descending order:

\[
\lambda_{i,1}\geq\lambda_{i,2}\geq\cdots\geq\lambda_{i,5}\geq0.
\]

The next window advances by one beat and shares four beats with the preceding
window. The current causal output becomes available when its fifth beat has
arrived.

The paper shows the uncentered `Q_i Q_i^T` construction. The published core
therefore does not silently mean-center or amplitude-normalize the waveforms.
Raw eigenvalues remain sensitive to gain, baseline, amplitude, and alignment.
Normalized eigenvalue fractions are exported only with an `exploratory_`
prefix and are not substituted for the published features.

## Per-beat output

Each waveform row records:

- timestamp track, sample, and time;
- capture boundaries and whether the complete 120-ms capture exists;
- waveform-array row index;
- expert symbol and candidate-to-expert timing error when available;
- independent-detector support fields when supplied;
- exploratory amplitude, peak-to-peak excursion, RMS, and maximum slope.

The waveform array itself is retained. Summary statistics are not a substitute
for visually verifying alignment and truncation.

## Per-five-beat output

Each feature row contains:

- all five timestamp samples and elapsed time;
- `lambda1` through `lambda5`;
- total eigenvalue energy;
- exploratory lambda-1 and lambda-3-to-5 energy fractions;
- complete-capture count and truncation flag;
- all five expert symbols, label transitions, and non-`N` count when known;
- detector disagreement, offsets, and Zhai correlation when supplied;
- explicit `seizure_probability_produced = false` and
  `artifact_label_produced = false` fields.

An expert symbol is context, not an output predicted by this branch. A `V`,
`L`, `R`, `F`, paced beat, or other abnormal morphology must not be deleted
merely because its eigenvalue pattern differs from normal beats.

## Fixed-window limitation

A fixed 120-ms capture can truncate a wide bundle-branch-block or ventricular
complex. It can also be misplaced when the input timestamp falls on another
QRS lobe. The branch currently reports only whether the array boundary was
available; it cannot prove that the complete physiological QRS was captured
because QRS onset/offset delineation has not yet been implemented.

Replacing the 120-ms capture with variable delineated windows is a separate
experimental branch, not a faithful Varon replication.

## Relationship to RR/HRV

RR/HRV and morphology may share a validated event/timestamp layer, but their
measurements remain separate:

```text
timestamp track
  |-- consecutive timestamps -> RR -> 100-RR HRV measurements
  `-- ECG around timestamps -> five-beat morphology eigenvalues
```

Fusion-ready rows preserve the timestamp source and join only same-track
measurements. They do not average detector timestamps, mix sources inside an
RR interval, or emit a seizure probability.

## Accuracy question

There are two different validations:

1. **Event accuracy:** candidate beats versus expert `.atr` annotations,
   reported as sensitivity, PPV, F1, and timing error.
2. **Feature fidelity:** candidate-timestamp eigenvalues versus eigenvalues
   obtained from expert timestamps on the same five consecutive beats.

A five-beat window is considered comparable only when all five expert beats
are matched and the candidate beats are consecutive in the candidate stream.
An extra or missed candidate invalidates that sequence instead of being hidden
by a nearest-neighbor match.

MIT--BIH does not contain seizure labels. Consequently, this audit cannot
measure seizure sensitivity, seizure PPV, or seizure timing.

## Current detector-track decision

The all-48 results are in
[`17_MORPHOLOGY_FIDELITY_AND_REVIEWER.md`](17_MORPHOLOGY_FIDELITY_AND_REVIEWER.md).
UNSW currently provides the greatest five-beat continuity; NeuroKit and Zhai
provide lower feature error on the windows they successfully cover. No result
supports declaring one universal morphology timestamp owner. All three remain
explicit audit tracks until representative patient ECG is manually annotated.

## Output artifacts

The NeuroKit--Zhai pipeline writes separate beat-context CSV files, 120-ms
waveform arrays, five-beat feature CSV files, and same-track fusion-ready
tables. The standalone benchmark additionally writes record-level, per-window,
and per-matched-beat audit tables for UNSW, NeuroKit, and Zhai.

Full QRS onset, Q, R/R-prime, S, and offset delineation remains `not_run`; it
must be validated against a delineation reference before entering this branch.
