"""Generate the four detailed, independently runnable ECG branch notebooks.

The generated notebooks orchestrate the tested package implementation instead
of copying and silently diverging from it.  Each notebook embeds a rendered
snapshot of every relevant source module, including SHA-256 hashes, so the
executed artifact preserves the exact implementation it used.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "feature_extraction"
NOTEBOOK_DIR = PROJECT / "notebooks" / "standalone_branches"


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip())


def walkthrough(
    title: str,
    why: str,
    chunks: list[tuple[str, str, str]],
    success: list[str],
    investigate: list[str],
):
    """Build the teaching note that precedes one executable code cell."""

    chunk_rows = "\n".join(
        f"| {index} | {name} | {action} | {reason} |"
        for index, (name, action, reason) in enumerate(chunks, start=1)
    )
    success_rows = "\n".join(f"- {item}" for item in success)
    investigate_rows = "\n".join(f"- {item}" for item in investigate)
    return md(
        f"""
        #### Code walkthrough — {title}

        **Why this cell exists.** {why}

        | Order | Code chunk | What executes | Why it is here |
        |---:|---|---|---|
        {chunk_rows}

        **Evidence that the cell ran correctly**

        {success_rows}

        **If that evidence is absent, investigate**

        {investigate_rows}

        The code cell immediately below is the executable source of these outputs.
        Read the chunks in table order; statements inside a loop or function belong
        to the chunk that introduces that block.
        """
    )


def code_walkthrough(branch: str, source: str):
    """Return a branch-aware explanation for every generated code cell."""

    if "search_roots =" in source and "OUTPUT_DIR" in source:
        return walkthrough(
            "imports, project discovery, and frozen run configuration",
            "Every later result depends on using the intended package, dataset, lead, units, and output folder.",
            [
                ("Imports", "Loads hashing, tables, numerical methods, plotting, NeuroKit, WFDB, and notebook display helpers.", "The notebook needs both computation and visible audit evidence."),
                ("Project discovery", "Searches the current directory and its parents for `pyproject.toml` and `src/ecg_cascade`, then places `src` on `sys.path`.", "The same notebook must run from the repository root or its own folder without importing an unrelated installed copy."),
                ("Output contract", "Creates one branch-specific directory under `outputs/standalone_notebooks`.", "CSV, JSON, and manifests remain separate and reproducible."),
                ("Input configuration", "Freezes record 100, lead MLII, start, duration, and patient identifier; the header assertion fails early if data are missing.", "A clinical reader must know exactly which signal produced the figures."),
                ("Display configuration", "Applies a consistent plot style and wide table display.", "Successful runs should be readable without hidden columns or default styling ambiguity."),
                ("Version printout", "Prints Python and dependency versions plus resolved paths.", "A future rerun can distinguish data or code changes from environment changes."),
            ],
            [
                "A compact version block is printed with Python, NumPy, pandas, SciPy, NeuroKit2, and WFDB versions.",
                "`Project` resolves to the repository's `feature_extraction` folder and `Output` resolves to this branch's folder.",
                "No import, missing-file, or path assertion error appears.",
            ],
            [
                "A wrong working directory or a missing `feature_extraction/pyproject.toml`.",
                "The MIT–BIH dataset path or the requested lead name.",
                "A notebook kernel that is not using the project virtual environment.",
            ],
        )

    if "segment = load_wfdb_segment" in source and "Input ECG" in source:
        return walkthrough(
            "load, validate, summarize, and preview the input ECG",
            "Algorithm outputs are interpretable only after the waveform, sampling rate, time coordinates, units, and finite-sample checks are visible.",
            [
                ("Loader import", "Imports the project's WFDB segment loader.", "This centralizes channel lookup and physical-unit conversion instead of duplicating raw WFDB calls."),
                ("Segment read", "Loads the configured lead and time range, then extracts samples and sampling frequency.", "Every sample index used later must map to a physical time."),
                ("Time axis", "Builds an absolute time vector from the segment start and sample rate.", "Plots and event tables then share the same coordinate system."),
                ("Assertions", "Requires one-dimensional, finite data and at least three seconds of signal.", "Downstream filters and interval calculations should fail visibly rather than accept malformed input."),
                ("Summary", "Prints source, lead, sampling rate, sample count, time span, and amplitude range.", "This is the run's input provenance and first unit sanity check."),
                ("Preview figure", "Plots the first 12 seconds of the unmodified physical ECG.", "A reviewer can see polarity, gain, baseline, and recognizable beats before any branch operates."),
            ],
            [
                "The summary reports 360 Hz, the requested duration, finite amplitude limits, and channel MLII.",
                "The figure shows a continuous ECG trace with no blank panel or clipped time axis.",
                "The x-axis is seconds and the y-axis explicitly says physical WFDB units.",
            ],
            [
                "Lead selection, start/duration bounds, or a corrupt/missing WFDB pair.",
                "NaN/Inf samples, an unexpectedly flat trace, or an amplitude scale inconsistent with the header.",
                "A wrong sampling rate, because every millisecond conversion below depends on it.",
            ],
        )

    if "SOURCE_FILES =" in source and "source_manifest" in source:
        return walkthrough(
            "embed exact implementation source and cryptographic hashes",
            "A notebook can otherwise become an attractive but stale copy of production code; this cell records precisely what implementation was executed.",
            [
                ("Source list", "Freezes the branch modules and scripts included in the appendix.", "The reader can audit the implementation boundary."),
                ("File loop", "Reads every file as UTF-8, counts lines, and calculates SHA-256.", "A one-character source change produces a different digest."),
                ("Rendered source", "Displays the complete Python text in the executed notebook.", "Every internal line remains reviewable without opening a second application."),
                ("Manifest export", "Writes `source_manifest.json` and displays its table.", "Automated checks can compare filenames, line counts, and hashes later."),
            ],
            [
                "Each expected module appears as a rendered code block with a 64-character SHA-256 digest.",
                "The final table contains one row per requested source file and positive line counts.",
                "`source_manifest.json` is saved in the branch output directory.",
            ],
            [
                "A source module that moved or is missing.",
                "An unexpected project root, which would hash the wrong files.",
                "Notebook size limits if a front end refuses to render the complete appendix.",
            ],
        )

    # RR / HRV branch.
    if branch == "rr_hrv" and "display(pd.Series(config.to_dict()" in source:
        return walkthrough(
            "freeze detector association and 100-RR feature parameters",
            "A tolerance, delay, window width, or causal filter change can materially change detector support and HRV values, so none may remain implicit.",
            [
                ("Configuration import", "Imports the immutable NeuroKit–Zhai configuration and cascade entry point.", "Both detector tracks use one recorded contract."),
                ("Orientation", "Keeps the original lead orientation and disables the separate inverted-context experiment.", "This run should not silently choose polarity from its own output."),
                ("Detector timing", "Sets NeuroKit's strict 300-ms delay and two association tolerances.", "Minimum separation affects double detections; matching affects context and audit only."),
                ("HRV construction", "Requires exactly 100 RR intervals and a trailing seven-RR median.", "These values implement the documented Jeppesen-style continuous feature construction."),
                ("Reliability policy", "Requires complete RR support for the reliability flag.", "A continuous value can remain defined while its support context is recorded separately."),
                ("Configuration display", "Shows every stored parameter.", "The PI can see the complete frozen contract before execution."),
            ],
            ["A two-column configuration table appears and includes 100 RR, median width 7, and 50-ms support tolerance.", "No timestamp-owner or classifier setting is introduced."],
            ["A changed default hidden outside the printed configuration.", "Confusing support tolerance with expert-audit tolerance.", "Treating `reliable_hrv_coverage=1.0` as a clinical validity threshold."],
        )

    if branch == "rr_hrv" and "tracks = {" in source and "detector_summary" in source:
        return walkthrough(
            "run the two independent detector-to-HRV tracks",
            "The central scientific question is what each timestamp owner produces before any fusion decision.",
            [
                ("Cascade call", "Processes the same ECG through NeuroKit and the Zhai reimplementation with the frozen configuration.", "Direct comparison requires identical input and scope."),
                ("Track dictionary", "Names the two independent results without selecting or averaging them.", "Each RR series must preserve single-detector timestamp ownership."),
                ("Detector summary", "Collects event, interval, support, defined-feature, and reliable-feature counts.", "Counts reveal missing beats, extras, warm-up, and support loss."),
                ("Association display", "Shows one-to-one agreement at the audit tolerance.", "Agreement is evidence about correspondence, not proof of correct physiology."),
            ],
            ["Both rows contain positive event and RR counts.", "At least one 100-RR feature row is defined for each track.", "The association table is finite and does not replace either timestamp list."],
            ["Detector failure or an ECG segment too short to reach 100 RR intervals.", "Unexpected event-count differences after a dependency/configuration change.", "Any code that merges or moves timestamps during association."],
        )

    if branch == "rr_hrv" and "Independent R-peak tracks" in source:
        return walkthrough(
            "overlay each detector's R anchors on the same ECG",
            "A numerical event count cannot reveal whether a marker sits on a QRS, T wave, or noise deflection.",
            [
                ("Figure layout", "Creates one shared-time panel per detector for the first 15 seconds.", "Separate panels avoid marker occlusion while preserving visual comparability."),
                ("ECG trace", "Plots the identical raw ECG in both panels.", "Differences must come from timestamps, not different preprocessing displays."),
                ("Time conversion", "Converts detector sample indices to absolute seconds.", "WFDB annotations and later tables use the same coordinate."),
                ("Marker interpolation", "Places colored points at each detected time using the raw signal amplitude.", "Markers visibly land on waveform deflections."),
                ("Labels", "Adds detector legends and shared units.", "The screenshot can be explained without reading variable names."),
            ],
            ["Two aligned panels appear, each with markers close to visible QRS complexes.", "The x-axes span 0–15 s and the detector colors/legends are distinct."],
            ["A sample-to-time conversion error if markers are horizontally shifted.", "Incorrect channel polarity or a detector reacting repeatedly to non-QRS waves.", "A blank marker set caused by a failed detector."],
        )

    if branch == "rr_hrv" and "rr_check =" in source:
        return walkthrough(
            "inspect event and RR tables and recompute every interval",
            "RR values must be direct differences of consecutive timestamps from one detector; otherwise all HRV features are scientifically invalid.",
            [
                ("Track selection", "Chooses the NeuroKit lane for a transparent hand audit while leaving Zhai intact.", "One lane is enough to demonstrate the invariant."),
                ("Table display", "Shows the first event rows and RR rows.", "Identifiers, times, support, and reason fields are visible."),
                ("Independent recomputation", "Calculates `diff(peak_samples) * 1000 / fs`.", "This is the simplest reference equation for RR milliseconds."),
                ("Exact comparison", "Uses NumPy testing to compare every recomputed value with the stored table.", "A silent row shift or cross-track interval fails immediately."),
            ],
            ["The event and interval previews share coherent indices/times.", "The final printed sentence confirms every NeuroKit RR interval passed the direct difference check."],
            ["Off-by-one event/interval alignment.", "A milliseconds-versus-seconds unit error.", "Any accidental interval built across two different detector tracks."],
        )

    if branch == "rr_hrv" and "RR tachograms" in source:
        return walkthrough(
            "show RR, causal filtering, and mature J1/J2 outputs over time",
            "This figure connects detector timestamps to the time series that a non-coder will actually interpret.",
            [
                ("Two-panel canvas", "Allocates a tachogram panel and a mature-feature panel with shared time.", "Raw timing and derived values can be compared vertically."),
                ("Raw RR", "Plots each interval endpoint against its RR duration.", "Extras, misses, and ectopy appear as abrupt interval changes."),
                ("Causal median-7", "Overlays the current-plus-previous-six RR median.", "The reader can see what short spikes the documented filter suppresses."),
                ("Defined mask", "Plots J1/J2 only after their 100-RR prerequisite is satisfied.", "Warm-up is honest missingness, not zero."),
                ("Legends and units", "Names both tracks and both products.", "The screenshot is self-contained."),
            ],
            ["The upper panel contains dots plus smoother median curves for both tracks.", "The lower panel begins later because 100 RR intervals are required.", "No early J1/J2 values are fabricated."],
            ["A flat or explosive RR series indicating missed/extra peaks.", "A non-causal centered median if changes appear before their inputs.", "Feature values appearing before the 100th RR row."],
        )

    if branch == "rr_hrv" and "poincare_axes" in source:
        return walkthrough(
            "hand-check one 100-RR Poincaré window",
            "SD1 and SD2 are geometric quantities; recomputing and plotting one complete window makes the equations tangible and auditable.",
            [
                ("Mature row selection", "Finds the latest row where the 100-RR feature is defined.", "It avoids a warm-up row with intentional NaNs."),
                ("Window slice", "Extracts exactly the current and preceding 99 RR intervals.", "The published feature family has a fixed 100-interval context."),
                ("Independent axes", "Calls the small Poincaré reference function and compares SD1/SD2 with stored columns.", "A window-boundary or axis-definition error becomes an assertion failure."),
                ("Scatter plot", "Plots each RR against its successor plus the identity line.", "Width across and length along the identity direction explain SD1/SD2 visually."),
                ("Feature row", "Displays CSI, modified CSI, slope, J1/J2, and reliability beside the plot.", "Geometry, derived products, and context are linked at one timestamp."),
            ],
            ["A 99-point Poincaré cloud appears with finite SD1 and SD2 in the title.", "The stored/direct comparison passes and the displayed row says `feature_defined=True`."],
            ["Fewer than 100 intervals, an empty defined-row set, or a slice boundary error.", "SD1/SD2 swapped or calculated from filtered data in the raw path.", "Interpreting the identity line as a clinical normal range."],
        )

    if branch == "rr_hrv" and "CASE_RECORD" in source:
        return walkthrough(
            "reproduce record 113's T-wave double-detection case",
            "The earlier audit showed that detector disagreement can arise on a clean-looking physiological wave, so reliability cannot be relabeled as artifact.",
            [
                ("Case scope", "Loads record 113 MLII from 893–901 s, the audited strip with repeated broad post-QRS waves.", "Using the same local data makes the report's claim reproducible."),
                ("Independent rerun", "Runs the same NeuroKit–Zhai cascade on only that strip.", "The case is generated by current code, not pasted as an unexplained image."),
                ("Expert references", "Loads `.atr` beat annotations and keeps only events inside the strip.", "Expert beat marks distinguish extra detector events from genuine annotated beats."),
                ("Count audit", "Compares NeuroKit, Zhai, and expert event counts and asserts the expected failure pattern.", "A dependency change that removes or reverses the example will fail visibly."),
                ("Annotated screenshot", "Plots the ECG with hollow expert diamonds, NeuroKit crosses, and Zhai markers.", "The broad waves receiving extra NeuroKit markers can be inspected directly."),
            ],
            ["Eight expert beats and approximately eight Zhai events are shown, while NeuroKit reports roughly twice as many events.", "Extra NeuroKit crosses recur about 0.3–0.36 s after QRS complexes on broad post-QRS waves.", "The assertions and printed `Case counts` complete without error."],
            ["Wrong record/channel/time range.", "A changed detector version or minimum-delay setting.", "Calling the broad wave definitively a T wave or calling the strip artifact—the report treats that mechanism as a supported inference, not annotation truth."],
        )

    if branch == "rr_hrv" and "exported = {}" in source:
        return walkthrough(
            "export independent event, RR, and HRV tables",
            "A final matrix is not auditable unless the intermediate timestamp and interval ownership tables remain available.",
            [
                ("Track loop", "Visits NeuroKit and Zhai separately.", "No output filename can hide which detector owns the timestamps."),
                ("Table map", "Assigns distinct event, RR, and feature CSV files.", "Reviewers can trace a feature row back to its endpoints."),
                ("CSV write", "Writes each table without a synthetic index and records its shape.", "Row/column counts provide a quick completeness check."),
                ("Configuration write", "Saves the same parameter dictionary shown earlier.", "Data and settings travel together."),
            ],
            ["Six CSV files plus `configuration.json` exist.", "The displayed shape table has nonzero rows and columns for both tracks."],
            ["An empty table caused by a detector or warm-up failure.", "Overwriting one track with the other's filename.", "A configuration file inconsistent with the printed settings."],
        )

    if branch == "rr_hrv" and "PASS — RR/HRV" in source:
        return walkthrough(
            "enforce notebook-level RR/HRV invariants",
            "A figure can look plausible while timestamps are unordered, intervals are non-positive, or classifier outputs have leaked into preprocessing.",
            [
                ("Ordering", "Requires strictly increasing detector sample indices.", "RR intervals need chronological events."),
                ("Physiology-independent arithmetic", "Requires positive RR durations and at least one defined mature feature.", "These are construction invariants, not clinical thresholds."),
                ("Context schema", "Requires defined/reliable/coverage columns and unmoved detector timestamps.", "Measurement and reliability stay separate."),
                ("Scope guard", "Confirms seizure and artifact classifiers were not applied.", "This notebook remains preprocessing-only."),
            ],
            ["The cell ends with the green-path sentence `PASS — RR/HRV notebook completed all acceptance checks.`"],
            ["The first assertion named in the traceback; it identifies the violated invariant.", "A detector/configuration change before weakening any check.", "Any new probability or clinical-label column, which belongs in a separately validated model."],
        )

    # Morphology branch.
    if branch == "morphology" and "NeuroKit anchors:" in source:
        return walkthrough(
            "detect one anchor track and reuse it across morphology experiments",
            "Window-method comparisons are only fair when their R anchors are identical.",
            [
                ("Cascade import/config", "Runs the current two-track detector pipeline without inverted-context selection.", "Anchor provenance stays explicit."),
                ("Track selection", "Uses the NeuroKit lane for this demonstration and retains its events.", "Every morphology row can be tied back to a named timestamp owner."),
                ("Fixed control", "Retrieves the already calculated published 120-ms Varon result.", "The unchanged control anchors all personalized-window comparisons."),
                ("Count print", "Reports anchor and five-beat-window counts.", "A short or failed run is obvious before expensive comparisons."),
            ],
            ["The printout contains more than 100 anchors and a nonzero fixed-Varon window count.", "All later methods receive the same `peaks` array."],
            ["Changing anchors between methods.", "Too few beats for five-beat windows.", "Treating the selected demonstration track as a validated final owner."],
        )

    if branch == "morphology" and "calibration = pf.loc" in source:
        return walkthrough(
            "derive explicit automatic QRS boundary calibration examples",
            "Personalized crops need measurable pre-R and post-R extents; using unnamed heuristics would make the comparison irreproducible.",
            [
                ("Delineation", "Runs the prominence P–QRS–T extractor on the shared anchors.", "It supplies both interpretable morphology and candidate QRS boundaries."),
                ("Eligibility filter", "Keeps the first 20 complete, physiologically ordered QRS complexes.", "Incomplete or misordered landmarks must not calibrate a crop."),
                ("Boundary arithmetic", "Converts onset-to-R, R-to-offset, and total QRS sample distances to milliseconds.", "Personalized rules operate in physical time, not dataset-specific samples."),
                ("Assertions and summaries", "Requires 20 positive durations and displays distributions plus delineation coverage.", "Bad landmark order or insufficient calibration fails before morphology extraction."),
            ],
            ["The duration table shows 20 positive observations with plausible medians and ranges.", "The PQRST summary reports complete QRS and ordered-beat counts."],
            ["Automatic prominence failures, especially P/T endpoints.", "A sign error in onset/R/offset subtraction.", "Mistaking automatic calibration boundaries for manual ground truth."],
        )

    if branch == "morphology" and "varon_methods =" in source:
        return walkthrough(
            "run four Varon crop experiments as separate lanes",
            "The project must learn which crop adds value; putting near-duplicate eigenvalue sets into one matrix would hide the ablation.",
            [
                ("Imports", "Loads mean-width, required-width, and asymmetric extractors.", "Each method has a distinct geometric rule."),
                ("Mean symmetric", "Uses the arithmetic mean total QRS duration centered on R.", "This reproduces the original personalization proposal, including its known limitation."),
                ("p95 symmetric", "Uses the 95th percentile of the width required to cover both sides.", "It is the preferred one-number symmetric challenger from held-out boundary tests."),
                ("p95 asymmetric", "Calibrates pre-R and post-R 95th percentiles separately with an explicit minimum sample size.", "R is often not centered within QRS."),
                ("Named result map", "Keeps fixed, mean, symmetric-p95, and asymmetric-p95 outputs distinct.", "No matrix can accidentally imply they are four independent biological signals."),
                ("Summary", "Reports realized crop, beat/window counts, defined Gram cores, and support context.", "Coverage differences are visible before feature comparison."),
            ],
            ["Four method rows appear with nonzero beats/windows and defined five-eigenvalue cores.", "Pre/post widths differ for the asymmetric method and remain equal for symmetric methods."],
            ["A fallback caused by too few calibration beats.", "Crop boundaries outside the signal.", "Claims of seizure superiority—the case only demonstrates extraction and boundary behavior."],
        )

    if branch == "morphology" and "Visualize the same five-beat window" in source:
        return walkthrough(
            "compare the same five beats under four capture rules",
            "A width number is abstract; aligned waveforms show exactly which pre/post-R signal each rule includes.",
            [
                ("Window choice", "Selects the same five-beat endpoint across all methods.", "The visual comparison controls for beat identity."),
                ("Waveform slice", "Uses each result's start/end beat indices to retrieve its five rows.", "Displayed beats are exactly those used for the Gram matrix."),
                ("Method time axis", "Builds a separate pre-R/post-R millisecond axis from realized sampling boundaries.", "Different crop widths are plotted honestly."),
                ("Overlay", "Draws five waveforms and a dashed R=0 reference in each panel.", "Shape consistency and included margins are immediately visible."),
            ],
            ["A 2×2 figure appears with five traces in every panel.", "All R anchors align at 0 ms while panel widths differ according to the method summary."],
            ["Comparing different beat indices across methods.", "Using requested instead of realized sample widths.", "Reading display overlap as proof that eigenvalues improve seizure discrimination."],
        )

    if branch == "morphology" and "direct_eigenvalues" in source:
        return walkthrough(
            "reconstruct the published five-beat Gram eigenspectrum",
            "This is the mathematical core of the Varon lane and should be independently reproducible from the displayed five waveforms.",
            [
                ("Five-beat matrix", "Stacks the selected fixed-control waveforms as rows of Q.", "The published construction operates on five captured QRS beats."),
                ("Gram matrix", "Calculates `Q @ Q.T`.", "Pairwise inner products encode shared waveform energy and similarity."),
                ("Symmetric eigensolver", "Numerically symmetrizes the Gram matrix, computes eigenvalues, and reverses them to descending order.", "The five lambdas form a stable ordered spectrum."),
                ("Stored-value test", "Compares direct eigenvalues with the branch feature row.", "Any normalization, centering, or ordering drift fails."),
                ("Heatmap and bars", "Shows beat-to-beat Gram structure and the five energy modes.", "The PI can connect the matrix to the numeric columns."),
            ],
            ["The heatmap is 5×5 and symmetric.", "Exactly five descending lambda bars appear.", "The direct/stored NumPy assertion passes."],
            ["Using columns instead of rows for beats.", "Unreported centering or normalization.", "Negative eigenvalues larger than tiny floating-point noise, which could indicate a malformed Gram matrix."],
        )

    if branch == "morphology" and "extract_patient_template_morphology" in source and "evaluation_index" not in source:
        return walkthrough(
            "build a fixed initial patient/lead template bank and score later beats",
            "Varon measures five-beat energy geometry; template features answer the different question 'how unlike this patient's earlier beats is the current cycle?'.",
            [
                ("Extractor import", "Loads the tested patient-template branch.", "The notebook orchestrates package code rather than maintaining a second implementation."),
                ("Template call", "Uses the first 20 complete cycles as calibration and later cycles as evaluation.", "Causal separation prevents a future beat from defining its own normal template."),
                ("Summary", "Displays calibration, eligibility, and availability counts.", "A missing or contaminated template bank is visible."),
                ("Feature preview", "Shows correlation, normalized error, raw residuals, derivative-DTW, and previous-beat change for evaluation rows.", "Each distance family retains its own meaning and units."),
            ],
            ["Twenty calibration members are reported and later evaluation rows exist.", "Evaluation feature values are finite and the preview distinguishes calibration from eligible rows."],
            ["Too few complete cycles.", "Calibration rows leaking into evaluation statistics.", "Calling template novelty a diagnosis; gain, ectopy, lead movement, and delineation error can all contribute."],
        )

    if branch == "morphology" and "matched_template =" in source:
        return walkthrough(
            "visualize one evaluation cycle, its closest fixed template, and residual",
            "Correlation and RMSE become understandable when the two normalized shapes and their pointwise difference are shown.",
            [
                ("Evaluation choice", "Selects a later eligible cycle and reads its matched-template index.", "The example cannot come from the calibration bank."),
                ("Normalized overlay", "Plots the fixed template and evaluation waveform over 0–100% of the resampled cycle.", "Timing is comparable despite different raw RR durations."),
                ("Residual", "Subtracts template from evaluation and plots the difference around zero.", "Residual amplitude shows where shape disagreement occurs."),
            ],
            ["The upper traces broadly overlap and the lower residual oscillates around zero.", "Labels clearly distinguish fixed template from evaluation beat."],
            ["An evaluation index outside the eligible set.", "A residual calculated in the opposite direction only if downstream sign matters.", "Confusing normalized residual with raw amplitude residual."],
        )

    if branch == "morphology" and "One automatically delineated P" in source:
        return walkthrough(
            "annotate one P–QRS–T complex and display interpretable morphology columns",
            "Eigenvalues and template distances need interpretable companions such as QRS duration, amplitudes, area, and polarity.",
            [
                ("Complete-beat choice", "Selects a beat with all P/QRS/T landmarks in physiological order.", "The example should not hide missing endpoints."),
                ("Landmark dictionary", "Converts named sample columns to integer indices.", "Every plotted marker is traceable to the exported table."),
                ("Local crop", "Adds 80-ms context around P onset and T offset.", "The full complex remains visible without plotting minutes of ECG."),
                ("Colored annotations", "Groups P, QRS, and T landmarks with distinct colors and labels.", "The screenshot teaches which boundaries generate each interval."),
                ("Measurement row", "Displays durations, amplitudes, area, polarity, and completeness flags.", "Visual and numeric morphology are linked for the same beat."),
            ],
            ["Nine ordered markers appear on one waveform.", "The feature row has finite QRS/P/QT values and both completeness/order flags are true."],
            ["Marker order violations or edge-clipped beats.", "Baseline definition when interpreting amplitudes/area.", "Treating automatic landmarks as cardiologist references."],
        )

    if branch == "morphology" and "CASE_SCREENSHOT" in source:
        return walkthrough(
            "display the prior record 207 morphology/polarity reviewer case",
            "Pooled averages hid a record where changing bundle-branch morphology and polarity produced large timestamp and five-beat-coverage errors.",
            [
                ("Evidence paths", "Locates the saved reviewer screenshot and all-48 morphology summary.", "The case is tied to reproducible repository artifacts."),
                ("Existence checks", "Fails if either evidence artifact is missing.", "A notebook should never render a broken or substituted case silently."),
                ("Case metrics", "Loads record 207 rows for the detector/feature-fidelity measures.", "The screenshot is accompanied by quantitative context."),
                ("Screenshot display", "Embeds the full multi-branch reviewer view in the executed notebook.", "The user requested a visible correct-run example for PI demonstration."),
            ],
            ["The displayed case table includes record 207 and the screenshot shows ECG, four marker tracks, five-beat lambda energy, and RR.", "Expert misses and morphology/polarity changes remain visible rather than collapsed into one score."],
            ["A stale screenshot from another record/channel.", "Missing expert `.atr` context.", "Interpreting display-normalized lambda fractions as the raw eigenvalue columns used by the extractor."],
        )

    if branch == "morphology" and "exports = {}" in source:
        return walkthrough(
            "export each morphology family without premature combination",
            "Ablation matrices require separable files for fixed/personalized Varon, templates, and PQRST measurements.",
            [
                ("Varon loop", "Writes one feature table per crop rule.", "Near-duplicate eigenvalue sets remain named experiments."),
                ("Template export", "Writes the per-cycle patient-template measurements.", "Novelty features stay distinct from five-beat eigenvalues."),
                ("PQRST export", "Writes interpretable landmarks and measurements.", "Durations/amplitudes can be audited independently."),
                ("Row-count display", "Reports each output's length.", "Missing families or unexpected row loss are immediately visible."),
            ],
            ["Six separate CSV files appear and each displayed row count is positive."],
            ["A shared filename overwriting another method.", "Combining columns before a held-out ablation.", "Different row granularities—five-beat windows versus single beats/cycles—before joining."],
        )

    if branch == "morphology" and "PASS — all morphology" in source:
        return walkthrough(
            "enforce morphology availability and scope guards",
            "The notebook must prove all requested families executed while preventing accidental seizure/artifact labels.",
            [
                ("Varon checks", "Requires a defined published core, lambda1–lambda5, and false classifier flags for every crop.", "Each experiment must contain the same mathematical core and remain preprocessing-only."),
                ("Template checks", "Requires eligible later cycles and finite correlations.", "A template bank alone is not enough; it must score held-out-in-time cycles."),
                ("PQRST checks", "Requires at least one complete QRS and no seizure output.", "Interpretable morphology must actually be available."),
            ],
            ["The cell ends with `PASS — all morphology implementations executed and retained separate outputs.`"],
            ["The named method/feature in the first failed assertion.", "Calibration completeness or crop bounds.", "Any new label/probability field added without a separately validated model."],
        )

    # Signal-quality branch.
    if branch == "signal_quality" and "UV_PER_INPUT_UNIT" in source and "other_lead_peaks" in source:
        return walkthrough(
            "declare units and construct every SQI prerequisite",
            "Amplitude SQIs, cross-lead agreement, detector agreement, and baseline estimates are meaningless if their units and event sources are implicit.",
            [
                ("Header audit", "Reads WFDB channel names/units and requires MLII to be stored in mV.", "The following µV conversion must match metadata."),
                ("Unit/mains constants", "Sets 1000 µV per input unit and an explicit 60-Hz demo assumption.", "Amplitude and powerline features need physical metadata."),
                ("Second lead", "Loads synchronized V5 with matching rate and length.", "Cross-lead iSQI compares events; it does not average waveforms."),
                ("Detector pair", "Creates NeuroKit and Zhai event arrays on MLII and NeuroKit events on V5.", "bSQI/qSQI/iSQI require named, independent inputs."),
                ("QRS onsets", "Runs prominence delineation and retains finite integer onset samples.", "Galeotti baseline anchors require QRS timing."),
                ("Count print", "Reports prerequisite event availability.", "Missing inputs are visible before the SQI table."),
            ],
            ["Header units print MLII/V5 as mV and all four event counts are positive.", "The synchronization assertions pass."],
            ["A dataset stored in V, µV, or ADC counts—change the conversion explicitly.", "Wrong mains-region metadata.", "Unsynchronized lead segments or missing QRS onset landmarks."],
        )

    if branch == "signal_quality" and "clean_quality = extract_signal_quality_window" in source:
        return walkthrough(
            "calculate the complete conditional SQI vector for one window",
            "One transparent window demonstrates both calculated values and honest missing-prerequisite reasons.",
            [
                ("Window slice", "Selects 20–30 s of MLII.", "A fixed 10-s case is easy to inspect and matches common SQI windows."),
                ("Local-event helper", "Filters global samples to the window and subtracts its start index.", "Feature code expects window-relative coordinates."),
                ("SQI extraction", "Supplies units, two detectors, QRS onsets, another lead, mains metadata, and exact-flat tolerance.", "Conditional features calculate only when their prerequisites exist."),
                ("Three-part table", "Aligns value, availability, and reason for every named feature.", "A NaN is never left unexplained."),
                ("Availability count", "Prints calculated versus total features.", "The reader sees schema completeness separately from numeric availability."),
            ],
            ["A full feature-indexed table appears with value/available/reason columns.", "Unavailable device-specific or grouped-template features state why instead of becoming zero."],
            ["Incorrect local event coordinates.", "Units or mains frequency omitted for features that require them.", "Interpreting an availability flag as a good/bad quality decision."],
        )

    if branch == "signal_quality" and "variants = {" in source:
        return walkthrough(
            "create controlled degradations and recompute SQIs",
            "A teaching notebook should show how each measurement responds before anyone proposes a classifier or threshold.",
            [
                ("Deterministic variants", "Creates clean, baseline-wander, high-frequency-noise, flatline, and clipped copies; the random generator has a fixed seed.", "Each corruption has a known mechanism and reproducible waveform."),
                ("Detector helper", "Runs NeuroKit and Zhai afresh on each variant.", "Detector-agreement SQIs should respond to the degraded signal, not stale clean events."),
                ("Quality loop", "Extracts the same feature schema for every condition.", "Only the waveform condition changes."),
                ("Comparison table", "Selects integrity, spectral, agreement, and template metrics.", "The PI sees complementary response types rather than one magic score."),
            ],
            ["Five condition rows appear with the same metric columns.", "Flatline fraction rises for the flatline variant and amplitude limits change for clipping."],
            ["Random noise without a fixed seed.", "Reusing clean detector peaks on degraded signals.", "Calling the most responsive metric a validated artifact classifier."],
        )

    if branch == "signal_quality" and "normalized = matrix.copy" in source:
        return walkthrough(
            "render waveform and relative-SQI response screenshots",
            "The first plot shows what was changed; the second shows how differently scaled metrics react without pretending they share units.",
            [
                ("Waveform stack", "Plots all five 10-s signals on aligned axes.", "The corruption mechanism is visually verifiable."),
                ("Per-column normalization", "Maps each metric's finite values to 0–1 only within this diagnostic.", "µV, fractions, and kurtosis can be displayed together without numerical domination."),
                ("Constant handling", "Assigns 0.5 when a column has no range.", "A constant feature is visible and avoids division by zero."),
                ("Heatmap", "Displays condition-by-feature relative response with labeled axes and colorbar.", "Complementary patterns and non-responses are easy to teach."),
            ],
            ["A five-row waveform figure and a five-row heatmap appear.", "The heatmap title explicitly says it is not an artifact score."],
            ["Comparing colors across columns as absolute magnitude.", "A NaN-only metric, which should remain visibly unavailable.", "Using the normalized display values as exported scientific features."],
        )

    if branch == "signal_quality" and "NSTDB_CASE" in source:
        return walkthrough(
            "replay the prior NSTDB electrode-motion case study",
            "Synthetic examples are useful, but the repository also contains a controlled benchmark with real electrode-motion noise mixed into MIT–BIH ECG at known SNR levels.",
            [
                ("Evidence load", "Reads the saved window-level features and Spearman rank diagnostic from the prior NSTDB run.", "The notebook cites empirical repository outputs rather than retyping summary numbers."),
                ("Scope filter", "Keeps the 12 electrode-motion windows from records 118/119 and sorts by SNR.", "Clean-source rows and unrelated conditions do not enter the case plot."),
                ("Correlation table", "Shows how each SQI ranks with improving SNR.", "Positive and negative directions are both meaningful and must not be hidden."),
                ("SNR trajectories", "Plots selected features separately for the two source records.", "A pooled correlation can be checked against record-level paths."),
                ("Rank bar chart", "Colors positive versus negative rho and marks zero.", "The successful-run screenshot summarizes monotonicity without implying accuracy."),
                ("Assertions", "Checks 12 windows and the previously documented basSQI/rms directions.", "A stale or wrong evidence file fails visibly."),
            ],
            ["Twelve noisy windows are reported across +24 to −6 dB.", "basSQI has strong positive rho with improving SNR, while RMS/HF RMS have negative rho.", "Menon is visibly near non-monotonic and is not promoted to a veto."],
            ["Wrong evidence directory or columns.", "Treating SNR rank correlation as artifact-classifier sensitivity/specificity.", "Transferring a threshold from NSTDB to a different patient/device."],
        )

    if branch == "signal_quality" and "extract_trailing_signal_quality_windows" in source:
        return walkthrough(
            "calculate causal trailing SQI windows",
            "A real-time demonstration needs to show that each displayed row uses only signal at or before its endpoint.",
            [
                ("Trailing extractor", "Uses 10-s windows every 5 s with the full-record prerequisite arrays.", "Overlap gives a smooth demonstration while preserving a causal trailing definition."),
                ("Frame conversion", "Expands the result and converts exclusive end samples to seconds.", "Every quality row receives an auditable support time."),
                ("Selected trajectories", "Plots basSQI, qSQI, and template correlation over window endpoints.", "Spectral, detector, and waveform-consistency families can disagree visibly."),
            ],
            ["A nonempty table begins at the first complete 10-s window.", "Three labeled curves appear against exclusive window-end time."],
            ["A centered/future-looking window implementation.", "Event arrays not adjusted by the extractor to each window.", "Interpreting stable SQIs as proof of seizure absence."],
        )

    if branch == "signal_quality" and "single_window_quality_with_availability.csv" in source:
        return walkthrough(
            "export SQI values together with availability and reasons",
            "Downstream users need to distinguish 'feature is low' from 'feature could not be calculated'.",
            [
                ("Single-window export", "Writes values, booleans, and reasons in one indexed CSV.", "The complete conditional schema survives outside the notebook."),
                ("Diagnostic export", "Writes controlled-degradation and trailing-window tables.", "Static and temporal demonstrations remain reproducible."),
                ("Parameter export", "Saves the extractor's recorded parameters as JSON.", "Units, frequency, and algorithm choices travel with measurements."),
                ("Directory listing", "Prints filenames and byte sizes.", "Zero-byte or missing artifacts are immediately visible."),
            ],
            ["Three CSV files and one JSON file are listed with positive byte sizes."],
            ["Dropping the availability/reason columns.", "Writing display-normalized heatmap values instead of raw SQIs.", "An output directory from a different branch."],
        )

    if branch == "signal_quality" and "PASS — signal-quality" in source:
        return walkthrough(
            "enforce SQI availability, response, causality, and no-classifier scope",
            "The branch is intentionally a measurement vector; the acceptance cell makes that scientific boundary executable.",
            [
                ("Core availability", "Requires finite fraction, bSQI, qSQI, and cross-lead iSQI in the fully supplied demo.", "All major prerequisite paths must execute."),
                ("Known response", "Requires the inserted flatline to increase flat fraction.", "At least one controlled mechanism is checked directionally."),
                ("Trailing coverage", "Requires nonempty aligned trailing output.", "The causal time-series path must run."),
                ("Forbidden schema", "Rejects artifact/clean/seizure labels or probabilities.", "No unvalidated classifier can masquerade as preprocessing."),
            ],
            ["The cell ends with `PASS — signal-quality measurements executed without inventing an artifact classifier.`"],
            ["Missing prerequisites before loosening an availability assertion.", "Incorrect variant construction if flatline response fails.", "Any newly introduced hard decision field."],
        )

    # Conduction branch.
    if branch == "conduction" and "prominence = extract_prominence_morphology" in source:
        return walkthrough(
            "create named ventricular anchors and P/QRS/T landmarks",
            "Conduction intervals are differences between boundaries, so anchor and delineator provenance must precede every number.",
            [
                ("Detector run", "Builds the NeuroKit–Zhai cascade and selects the named NeuroKit lane for the demonstration.", "R anchors define beat identity and RR context."),
                ("Prominence delineation", "Finds P, QRS, and T landmarks on those anchors.", "PR/QRS/QT/JT require explicit endpoints."),
                ("Feature table", "Retains one row per anchor with values and completeness/order flags.", "Missing landmarks remain attached to their beat."),
                ("Summary preview", "Shows aggregate availability and the first rows.", "Coverage and schema can be checked before interval arithmetic."),
            ],
            ["The summary reports many anchors and complete QRS/PQRST rows.", "The preview contains sample indices, measurements, and validity flags."],
            ["A poor anchor track, especially on atrial or broad-QRS deflections.", "Missing P onset/T offset coverage.", "Treating the automatic delineator as manual interval truth."],
        )

    if branch == "conduction" and "Landmarks consumed by the conduction branch" in source:
        return walkthrough(
            "show exactly which landmarks feed the interval equations",
            "A non-coder can understand PR, QRS, QT, and JT only if their endpoints are visible on one beat.",
            [
                ("Valid-beat selection", "Chooses a complete, physiologically ordered PQRST row.", "No hidden interpolation fills absent endpoints."),
                ("Named coordinates", "Builds an integer sample dictionary for all nine landmarks.", "The plot and source table share exact indices."),
                ("Context crop", "Displays the waveform from just before P onset to just after T offset.", "The whole conduction/repolarization cycle is visible."),
                ("Markers/labels", "Places and names every landmark at its raw amplitude.", "The screenshot is an anatomy key for later formulas."),
            ],
            ["Nine labels occur in physiological left-to-right order on one ECG cycle."],
            ["An edge beat or invalid landmark order.", "Labels placed on a filtered trace while sample indices refer to raw ECG.", "Overlapping annotations that obscure P or T endpoints."],
        )

    if branch == "conduction" and "timing = extract_conduction_timing" in source:
        return walkthrough(
            "calculate frozen beatwise intervals, QTc alternatives, and 30-beat summaries",
            "The branch needs both interpretable current-beat values and trailing variability, each with explicit missingness.",
            [
                ("Timing extractor", "Consumes the PQRST table at the known sampling rate with a 30-beat window and 80% minimum valid fraction.", "Parameters controlling availability are explicit."),
                ("Beat table", "Returns RR/HR context, raw intervals, QTc alternatives, differences, and defined flags.", "No single QT correction is silently privileged."),
                ("Rolling table", "Returns causal 30-beat summaries and variability measures.", "Longer context is kept separate from beatwise measurements."),
                ("Summary/preview", "Displays counts and selected columns for the first beats.", "Warm-up and missing endpoints are visible."),
            ],
            ["Beat rows equal anchor rows and finite PR/QRS/QT/JT values appear after the first beat where applicable.", "The summary reports nonzero rolling-feature availability."],
            ["Sampling-rate or endpoint-unit errors.", "Missing preceding RR for the first beat—this is expected.", "Treating QTc alternatives as diagnoses or interchangeable numbers."],
        )

    if branch == "conduction" and "direct = {" in source:
        return walkthrough(
            "independently verify interval and QT-correction equations",
            "A direct one-beat calculation catches sign, endpoint, unit, and row-alignment errors that plots may hide.",
            [
                ("Complete-row choice", "Selects a beat where all raw intervals are defined.", "The reference calculation needs every endpoint."),
                ("Direct differences", "Computes PR, QRS, QT, and JT from sample differences times 1000/fs.", "This is the physical definition of each interval."),
                ("Stored comparison", "Checks every direct result against the branch table.", "A swapped endpoint or offset fails immediately."),
                ("QTc checks", "Uses preceding RR seconds to recompute Bazett and Fridericia values.", "The denominator unit and exponent are explicitly tested."),
                ("Comparison table", "Shows direct and stored values side-by-side.", "The PI can verify exact equality visually."),
            ],
            ["The table's direct/stored columns match and the final sentence confirms equation checks passed."],
            ["Milliseconds versus seconds in RR.", "Using T peak instead of T offset for QT.", "Row mismatch between landmarks and beat features."],
        )

    if branch == "conduction" and "Named QT-correction alternatives" in source:
        return walkthrough(
            "plot raw intervals, QTc alternatives, and causal QT variability",
            "The figure shows that formula choice and landmark variability materially affect the curves; no correction should be hidden behind a generic `QTc` label.",
            [
                ("Raw intervals", "Plots PR, QRS, QT, and JT for every beat where defined.", "Different endpoints create distinct trajectories and scales."),
                ("QTc alternatives", "Plots Bazett, Fridericia, and Framingham with separate names.", "Rate correction is a modeling choice, not a universal transformation."),
                ("Defined-window mask", "Keeps only rows where the 30-beat QT summary meets its requirement.", "Warm-up/missingness are not plotted as zeros."),
                ("Variability curves", "Shows QT SD30 and RMSSD30 against time.", "The PI sees why endpoint error can inflate variability."),
            ],
            ["Three stacked panels appear; the variability panel begins only after sufficient history.", "QTc curves are similar but not identical."],
            ["Huge discontinuities caused by failed T offsets.", "A generic unlabeled QTc column.", "Interpreting automatic QT variability as validated physiology before the manual interval gate passes."],
        )

    if branch == "conduction" and "information = extract_conduction_information" in source:
        return walkthrough(
            "build five-beat descriptive and nine-beat QT context tables",
            "The literature supports raw PQRST timing dynamics and nine-beat QT context, but not an invented five-beat seizure score.",
            [
                ("Information extractor", "Requests trailing five-beat windows with at least four valid beats and no external eligibility column.", "The demo shows mechanics while recording that no artifact gate was applied."),
                ("Summary", "Reports beat, window, measurement, and availability counts.", "The long-table scale is transparent."),
                ("Five-beat preview", "Shows one row per measurement per endpoint with robust/descriptive statistics.", "Different intervals are not forced into one scalar."),
                ("Nine-beat preview", "Shows paired mean QT/RR and named corrections for complete nine-beat contexts.", "This preserves the Brotherstone-style context separately."),
            ],
            ["Nonempty long-format five-beat rows and complete nine-beat QT rows appear.", "Metadata/columns retain information-only and non-integration status."],
            ["Insufficient valid beats or an unintended eligibility filter.", "Calling five-beat variability a published Diab feature—the paper used ordered raw peak dynamics.", "Calling nine-beat QT context a reproduced seizure detector."],
        )

    if branch == "conduction" and "CASE_RECORD" in source:
        return walkthrough(
            "reproduce record 231's conduction-related false-detection strip",
            "The prior report showed that smaller physiological atrial deflections can be mistaken for ventricular beats; agreement/disagreement must not be collapsed into generic artifact.",
            [
                ("Case scope", "Loads record 231 MLII from 461.8–470.3 s.", "This is the same explanatory strip documented in the prior case report."),
                ("Detector rerun", "Runs NeuroKit and Zhai on the current local signal.", "The screenshot is produced by current branch code."),
                ("Expert beats", "Loads and time-filters `.atr` ventricular beat annotations.", "Annotated QRS events anchor the interpretation."),
                ("Count assertions", "Requires five expert beats, a matching-size Zhai lane, and extra NeuroKit events.", "The known regression pattern fails visibly if behavior changes."),
                ("Annotated plot", "Overlays expert, NeuroKit, and Zhai markers on the ECG.", "Smaller intervening deflections and the detector responses are directly visible."),
            ],
            ["Five expert and five Zhai events align with large QRS complexes, while NeuroKit also marks smaller intervening deflections.", "The printed case counts and assertions pass."],
            ["Calling every intervening deflection definitively a P wave—the report says that identity is physiologically plausible but not directly annotated.", "Using the case to declare Zhai universally superior.", "Allowing an extra ventricular timestamp to create false short RR intervals."],
        )

    if branch == "conduction" and "conduction_metadata.json" in source:
        return walkthrough(
            "export all conduction granularities and audit metadata",
            "Beatwise, 30-beat, five-beat-long, and nine-beat tables have different meanings and must remain distinguishable.",
            [
                ("Table map", "Names PQRST, beat, rolling30, information-window, and QT9 outputs.", "Every processing stage remains independently inspectable."),
                ("CSV loop", "Writes each table without an extra index.", "Schemas can be consumed by later ablations."),
                ("Metadata bundle", "Collects delineation, timing, information parameters, and diagnostics.", "Boundary method and availability rules travel with values."),
                ("Shape table", "Displays rows and columns for all exports.", "Empty or unexpectedly narrow tables are obvious."),
            ],
            ["Five CSV files plus `conduction_metadata.json` exist and every shape row is nonzero."],
            ["Joining different granularities without explicit keys.", "Dropping landmark provenance.", "Promoting information-only columns into a final matrix before validation."],
        )

    if branch == "conduction" and "PASS — conduction timing" in source:
        return walkthrough(
            "enforce conduction row alignment, availability, and non-clinical scope",
            "Interval arithmetic must stay aligned with anchors, and exploratory information must remain excluded from the final matrix until its measurement gate passes.",
            [
                ("Row alignment", "Requires equal anchor, PQRST, and beat-table lengths.", "Every interval row must refer to exactly one R anchor."),
                ("Availability", "Requires QRS, QT, rolling30 QT, and complete five-beat histories.", "All requested paths executed on the demo."),
                ("No clinical outputs", "Rejects seizure probabilities, clinical interpretations, abnormality labels, and final-matrix eligibility.", "The branch is measurement-only."),
            ],
            ["The cell ends with `PASS — conduction timing and information branches executed with all safety assertions intact.`"],
            ["The first failed row-count or availability check.", "Upstream P/T delineation quality.", "Any attempt to bypass the frozen non-integration flags."],
        )

    return walkthrough(
        "execute and audit this branch step",
        "This cell is part of the reproducible branch workflow and is documented here so no executable block appears without purpose or success criteria.",
        [
            ("Executable block", "Runs the statements exactly as shown in the code cell.", "The generated notebook keeps computation, output, and narrative adjacent."),
            ("Visible output", "Displays or saves the values produced by this step.", "The reader can verify progress without trusting hidden state."),
        ],
        ["The cell completes without an exception and produces the output described by the surrounding section."],
        ["The traceback and the inputs created by the immediately preceding cells.", "A changed package API or missing prerequisite."],
    )


def explain_all_code_cells(cells: list, branch: str) -> list:
    """Insert a detailed explanation immediately before every code cell."""

    explained = []
    for cell in cells:
        if cell.cell_type == "code":
            explained.append(code_walkthrough(branch, cell.source))
        explained.append(cell)
    return explained


def common_setup(branch_folder: str, duration_s: int = 120) -> list:
    return [
        md(
            f"""
            ## 1. Reproducible environment and data selection

            This notebook is **independent of the other notebooks**. It locates the
            project, imports the installed/tested branch package, selects one ECG
            channel explicitly, creates its own output directory, and records the
            runtime versions. The executable example uses MIT–BIH record 100,
            channel `MLII`, starting at 0 s for {duration_s} s.

            Change only the configuration cell below to use another WFDB record.
            For EDF data, use `ecg_cascade.edf.load_ecg_segment` after explicitly
            inspecting and choosing the ECG channel.
            """
        ),
        code(
            f"""
            from pathlib import Path
            import hashlib
            import json
            import platform
            import sys
            import warnings

            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            import scipy
            import neurokit2 as nk
            import wfdb
            from IPython.display import Code, Markdown, display

            # Locate feature_extraction regardless of whether execution starts from
            # the repository root, feature_extraction, or the notebook directory.
            search_roots = [Path.cwd().resolve(), *Path.cwd().resolve().parents]
            PROJECT = next(
                path for path in search_roots
                if (path / "pyproject.toml").is_file()
                and (path / "src" / "ecg_cascade").is_dir()
            )
            ROOT = PROJECT.parent
            SRC = PROJECT / "src"
            if str(SRC) not in sys.path:
                sys.path.insert(0, str(SRC))

            OUTPUT_DIR = PROJECT / "outputs" / "standalone_notebooks" / "{branch_folder}"
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

            DEMO_RECORD = (
                ROOT / "Datasets" / "mit-bih-arrhythmia-database-1.0"
                / "mit-bih-arrhythmia-database-1.0.0" / "100"
            )
            assert DEMO_RECORD.with_suffix(".hea").is_file(), DEMO_RECORD

            RECORD_PATH = DEMO_RECORD       # WFDB path without extension
            CHANNEL = "MLII"                # explicit lead choice
            START_S = 0.0
            DURATION_S = {duration_s}.0
            PATIENT_ID = "mitdb_100"

            plt.style.use("seaborn-v0_8-whitegrid")
            plt.rcParams.update({{
                "figure.figsize": (12, 4.5),
                "figure.dpi": 110,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.titleweight": "bold",
                "axes.labelcolor": "#304955",
                "text.color": "#203845",
            }})
            pd.set_option("display.max_columns", 120)
            pd.set_option("display.width", 180)

            print("Python:", platform.python_version())
            print("NumPy:", np.__version__, "Pandas:", pd.__version__, "SciPy:", scipy.__version__)
            print("NeuroKit2:", nk.__version__, "WFDB:", wfdb.__version__)
            print("Project:", PROJECT)
            print("Output:", OUTPUT_DIR)
            """
        ),
        code(
            """
            from ecg_cascade.wfdb_io import load_wfdb_segment

            segment = load_wfdb_segment(
                RECORD_PATH,
                channel=CHANNEL,
                start_s=START_S,
                duration_s=DURATION_S,
            )
            ecg = segment.samples
            fs = float(segment.sampling_rate_hz)
            time_s = segment.start_s + np.arange(ecg.size) / fs

            assert ecg.ndim == 1
            assert np.all(np.isfinite(ecg))
            assert ecg.size >= int(3 * fs)
            print({
                "source": segment.record_path,
                "channel": segment.channel_name,
                "sampling_rate_hz": fs,
                "samples": int(ecg.size),
                "start_s": segment.start_s,
                "duration_s": segment.duration_s,
                "minimum": float(ecg.min()),
                "maximum": float(ecg.max()),
            })

            preview = time_s < time_s[0] + 12
            fig, ax = plt.subplots(figsize=(13, 3.2))
            ax.plot(time_s[preview], ecg[preview], color="#0b9787", lw=1.0)
            ax.set(title="Input ECG — first 12 seconds", xlabel="Time (s)", ylabel="Amplitude (WFDB physical units)")
            plt.show()
            """
        ),
        md(
            """
            ### Reproducibility rules used throughout

            - No missing-sample imputation is performed.
            - The selected channel and physical-unit scale are shown explicitly.
            - Detector disagreement is context, not ground truth or artifact.
            - All rolling windows are trailing/causal where the implementation says so.
            - No notebook cell trains a seizure classifier or produces a clinical label.
            - Tables are exported separately so intermediate measurements remain auditable.
            """
        ),
    ]


def source_appendix(files: list[str]) -> list:
    rendered = repr(files)
    return [
        md(
            """
            ## Exact implementation appendix

            To make the executed notebook a durable audit artifact, this section
            embeds the **complete source text** of the modules used above and records
            a SHA-256 digest for each file. The algorithms are imported from these
            tested modules; they are not rewritten in a second, drifting notebook copy.
            """
        ),
        code(
            f"""
            SOURCE_FILES = {rendered}
            source_manifest = []
            for relative in SOURCE_FILES:
                path = PROJECT / relative
                text = path.read_text(encoding="utf-8")
                digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                source_manifest.append({{"file": relative, "sha256": digest, "lines": len(text.splitlines())}})
                display(Markdown(f"### `{{relative}}`  \\nSHA-256: `{{digest}}` · {{len(text.splitlines())}} lines"))
                display(Code(text, language="python"))

            source_manifest_path = OUTPUT_DIR / "source_manifest.json"
            source_manifest_path.write_text(json.dumps(source_manifest, indent=2), encoding="utf-8")
            display(pd.DataFrame(source_manifest))
            print("Saved:", source_manifest_path)
            """
        ),
    ]


def rr_hrv_notebook():
    cells = [
        md(
            r"""
            # Standalone RR / HRV preprocessing branch

            **Purpose.** Convert a verified single-lead ECG segment into independent
            R-peak timestamp tracks, RR intervals, heart rate, reliability context,
            and causal 100-RR Jeppesen measurements.

            **This notebook contains:** explicit configuration; input validation;
            NeuroKit and Zhai detector intermediates; detector association; RR
            construction; 7-RR causal median filtering; Poincaré SD1/SD2; CSI;
            modified CSI; heart-rate slope; J1/J2; reliability coverage; plots;
            exports; assertions; limitations; and complete source snapshots.

            **It does not:** choose a timestamp winner, remove intervals because two
            detectors disagree, label artifact, or predict seizures.

            ### Data flow

            ```text
            physical ECG
              ├─ NeuroKit gradient detector ─┬─ RR intervals ── 100-RR HRV
              │                              └─ reliability context versus Zhai
              └─ Zhai 2023 reimplementation ┬─ RR intervals ── 100-RR HRV
                                             └─ reliability context versus NeuroKit
            ```
            """
        ),
        *common_setup("rr_hrv", duration_s=120),
        md(
            r"""
            ## 2. Freeze the branch configuration

            The two timestamp sequences are calculated independently. Matching at
            ±50 ms supplies context only; it does not splice, average, or vote on
            timestamps. The HRV window contains exactly 100 consecutive RR intervals.

            For RR values \(RR_i\) in milliseconds:

            \[
            HR_i = \frac{60000}{RR_i}, \qquad
            SD1 = SD\left(\frac{RR_{i+1}-RR_i}{\sqrt{2}}\right), \qquad
            SD2 = SD\left(\frac{RR_{i+1}+RR_i}{\sqrt{2}}\right)
            \]

            \[
            CSI = \frac{SD2}{SD1}, \qquad
            ModCSI = \frac{4SD2_f^2}{SD1_f}, \qquad
            J1 = CSI\,|slope(HR_f)|, \qquad
            J2 = ModCSI\,|slope(HR_f)|
            \]

            The subscript \(f\) refers to the causal seven-RR median-filtered series.
            """
        ),
        code(
            """
            from ecg_cascade import NeuroKitZhaiConfig, run_neurokit_zhai_array

            config = NeuroKitZhaiConfig(
                processing_orientation="original",
                compute_inverted_context=False,
                neurokit_minimum_delay_ms=300.0,
                neurokit_minimum_delay_inclusive=False,
                support_tolerance_ms=50.0,
                audit_match_tolerance_ms=75.0,
                hrv_window_rr_intervals=100,
                causal_rr_median_width=7,
                reliable_hrv_coverage=1.0,
            )
            display(pd.Series(config.to_dict(), name="value").to_frame())
            """
        ),
        md("## 3. Run both timestamp tracks and inspect detector-level evidence"),
        code(
            """
            cascade = run_neurokit_zhai_array(
                ecg,
                fs,
                segment_start_s=segment.start_s,
                config=config,
                source_metadata={"record": str(RECORD_PATH), "channel": CHANNEL},
            )
            tracks = {
                "NeuroKit": cascade.selected.neurokit,
                "Zhai": cascade.selected.zhai,
            }
            detector_summary = pd.DataFrame({name: track.summary() for name, track in tracks.items()}).T
            display(detector_summary[[
                "event_count", "rr_interval_count", "supported_event_count",
                "supported_rr_count", "defined_hrv_rows", "reliable_hrv_rows"
            ]])
            display(pd.Series(cascade.selected.audit_agreement.to_dict(), name="detector association at 75 ms"))
            """
        ),
        code(
            """
            fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True, constrained_layout=True)
            view_start, view_end = 0.0, 15.0
            visible = (time_s >= view_start) & (time_s <= view_end)
            colors = {"NeuroKit": "#246d8f", "Zhai": "#d38232"}
            for ax, (name, track) in zip(axes, tracks.items()):
                ax.plot(time_s[visible], ecg[visible], color="#294651", lw=0.9)
                peak_times = segment.start_s + track.detector.peak_samples / fs
                shown = (peak_times >= view_start) & (peak_times <= view_end)
                y = np.interp(peak_times[shown], time_s, ecg)
                ax.scatter(peak_times[shown], y, s=28, color=colors[name], zorder=3, label=f"{name} R peaks")
                ax.legend(loc="upper right")
                ax.set_ylabel("ECG")
            axes[0].set_title("Independent R-peak tracks — same raw ECG")
            axes[-1].set_xlabel("Time (s)")
            plt.show()
            """
        ),
        md(
            """
            ## 4. RR construction and reliability context

            An RR interval joins consecutive timestamps **within the same detector
            track**. Unsupported intervals are retained; `rr_supported` and
            `reliability_reason` travel beside the measurement. This avoids turning
            detector disagreement into false physiological missingness.
            """
        ),
        code(
            """
            neurokit_track = cascade.selected.neurokit
            display(neurokit_track.events.head(8))
            display(neurokit_track.rr_intervals.head(8))

            rr_check = np.diff(neurokit_track.detector.peak_samples) * 1000.0 / fs
            np.testing.assert_allclose(rr_check, neurokit_track.rr_intervals["rr_ms"].to_numpy(float))
            print("Direct timestamp-difference check passed for every NeuroKit RR interval.")
            """
        ),
        code(
            """
            fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True, constrained_layout=True)
            for name, track in tracks.items():
                frame = track.features
                axes[0].plot(frame["end_time_s"], frame["rr_ms"], ".", ms=3, alpha=0.55, label=f"{name} raw RR")
                axes[0].plot(frame["end_time_s"], frame["rr_median7_ms"], lw=1.5, label=f"{name} causal median-7")
                defined = frame["feature_defined"]
                axes[1].plot(frame.loc[defined, "end_time_s"], frame.loc[defined, "j1_csi_x_slope"], lw=1.4, label=f"{name} J1")
                axes[1].plot(frame.loc[defined, "end_time_s"], frame.loc[defined, "j2_modcsi_filtered_x_slope"], lw=1.2, alpha=.8, label=f"{name} J2")
            axes[0].set(title="RR tachograms", ylabel="RR (ms)")
            axes[1].set(title="Defined 100-RR features", xlabel="Time (s)", ylabel="Feature value")
            for ax in axes: ax.legend(ncol=2, fontsize=8)
            plt.show()
            """
        ),
        md("## 5. Hand-check the latest 100-RR Poincaré window"),
        code(
            """
            from ecg_cascade.rr_hrv import poincare_axes

            hrv = neurokit_track.features
            last_defined = int(np.flatnonzero(hrv["feature_defined"].to_numpy(bool))[-1])
            first = last_defined - config.hrv_window_rr_intervals + 1
            rr100 = hrv.loc[first:last_defined, "rr_ms"].to_numpy(float)
            sd1_check, sd2_check = poincare_axes(rr100)
            row = hrv.loc[last_defined]
            np.testing.assert_allclose([sd1_check, sd2_check], [row.sd1_raw_ms, row.sd2_raw_ms])

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(rr100[:-1], rr100[1:], s=18, color="#6d58bd", alpha=.7)
            limits = [min(rr100) - 20, max(rr100) + 20]
            ax.plot(limits, limits, "--", color="#9aa7ad", lw=1)
            ax.set(xlim=limits, ylim=limits, xlabel=r"$RR_i$ (ms)", ylabel=r"$RR_{i+1}$ (ms)",
                   title=f"Latest 100-RR Poincaré window\\nSD1={sd1_check:.2f} ms, SD2={sd2_check:.2f} ms")
            plt.show()
            display(row[["end_time_s", "sd1_raw_ms", "sd2_raw_ms", "csi100", "modcsi100_filtered_ms",
                         "slope100_bpm_per_s", "j1_csi_x_slope", "j2_modcsi_filtered_x_slope",
                         "rr_reliability_coverage", "feature_defined", "feature_reliable"]])
            """
        ),
        md(
            """
            ## 6. Case study — record 113 shows why disagreement is not artifact

            The repository's prior [record-113 case report](../../reports/MITBIH_Record_113_TWave_Double_Detection_Case_Analysis.pdf)
            found a specific detector failure: NeuroKit matched almost every expert
            QRS but also placed 1,039 full-record extra markers on broad waves about
            306–356 ms after preceding expert beats. Their timing and shape are
            consistent with T-wave double detection, but the `.atr` file does not
            label T waves, so that mechanism remains a waveform-based inference.

            This matters because the signal was not thereby proven to be artifact.
            An RR-support failure can mean "the comparator saw an extra physiological
            wave," not "the electrode moved." The cell below reruns the current
            NeuroKit–Zhai branch on the report's 893–901 s strip and overlays expert
            beat annotations.

            Related evidence:

            - [Full RR/R-peak audit](../../reports/RR_HRV_Cascade_RPeak_Audit.pdf), especially the record-specific counterexamples.
            - Zhai et al. (2023), *Precise detection and localization of R-peaks from ECG signals*, [DOI 10.3934/mbe.2023848](https://doi.org/10.3934/mbe.2023848).
            - Jeppesen et al. (2025), phase-3 100-RR HRV method, [DOI 10.1016/j.ebiom.2025.105952](https://doi.org/10.1016/j.ebiom.2025.105952).
            """
        ),
        code(
            """
            from ecg_cascade.wfdb_io import load_wfdb_beat_annotations

            CASE_RECORD = (
                ROOT / "Datasets" / "mit-bih-arrhythmia-database-1.0"
                / "mit-bih-arrhythmia-database-1.0.0" / "113"
            )
            case_segment = load_wfdb_segment(
                CASE_RECORD, channel="MLII", start_s=893.0, duration_s=8.0
            )
            case_cascade = run_neurokit_zhai_array(
                case_segment.samples,
                case_segment.sampling_rate_hz,
                segment_start_s=case_segment.start_s,
                config=config,
                source_metadata={"record": "113", "case": "post-QRS double detections"},
            )

            case_reference_samples, case_reference_symbols = load_wfdb_beat_annotations(CASE_RECORD)
            case_reference_times_all = case_reference_samples / case_segment.sampling_rate_hz
            case_reference_keep = (
                (case_reference_times_all >= case_segment.start_s)
                & (case_reference_times_all < case_segment.start_s + case_segment.duration_s)
            )
            case_reference_times = case_reference_times_all[case_reference_keep]
            case_reference_symbols = np.asarray(case_reference_symbols)[case_reference_keep]

            case_nk_times = (
                case_segment.start_s
                + case_cascade.selected.neurokit.detector.peak_samples / case_segment.sampling_rate_hz
            )
            case_zhai_times = (
                case_segment.start_s
                + case_cascade.selected.zhai.detector.peak_samples / case_segment.sampling_rate_hz
            )
            case_counts = pd.Series({
                "expert_atr_beats": case_reference_times.size,
                "NeuroKit_events": case_nk_times.size,
                "Zhai_events": case_zhai_times.size,
            }, name="count")
            display(case_counts.to_frame())

            assert case_reference_times.size == 8
            assert case_nk_times.size > 1.8 * case_reference_times.size
            assert abs(case_zhai_times.size - case_reference_times.size) <= 1

            case_time = case_segment.start_s + np.arange(case_segment.samples.size) / case_segment.sampling_rate_hz
            fig, ax = plt.subplots(figsize=(14, 4.8))
            ax.plot(case_time, case_segment.samples, color="#244652", lw=1.0, label="raw MLII ECG")
            for reference_time in case_reference_times:
                ax.axvspan(reference_time + 0.28, reference_time + 0.38, color="#f4c96b", alpha=.16)
            ax.scatter(
                case_reference_times,
                np.interp(case_reference_times, case_time, case_segment.samples),
                marker="D", facecolors="none", edgecolors="#111111", s=58, linewidths=1.3,
                label="expert .atr beat",
            )
            ax.scatter(
                case_nk_times,
                np.interp(case_nk_times, case_time, case_segment.samples),
                marker="x", color="#d94f3d", s=52, linewidths=1.8,
                label="NeuroKit event",
            )
            ax.scatter(
                case_zhai_times,
                np.interp(case_zhai_times, case_time, case_segment.samples),
                marker="|", color="#7057b8", s=120, linewidths=2.2,
                label="Zhai event",
            )
            ax.set(
                title="Case study: MIT–BIH record 113 — repeated post-QRS NeuroKit events",
                xlabel="Time from recording start (s)", ylabel="MLII (mV)",
            )
            ax.legend(ncol=4, loc="upper center")
            plt.show()
            print("PASS — record 113 reproduces the audited extra-event pattern without calling it artifact.")
            """
        ),
        md(
            """
            ### How to read the successful-run screenshot

            1. Hollow black diamonds are expert-annotated beats and should sit on the
               large QRS complexes.
            2. Purple Zhai ticks should be close to those eight expert beats in this
               particular strip.
            3. Red NeuroKit crosses should appear both on QRS complexes and on many
               broad post-QRS waves. The pale amber bands mark the report's measured
               280–380 ms post-QRS region as a visual guide, not a decision threshold.
            4. The count table should show 8 expert beats, about 8 Zhai events, and
               about 17 NeuroKit events. That is a detector-specific case result—not
               a general ranking and not an artifact label.
            """
        ),
        md("## 7. Export auditable branch tables"),
        code(
            """
            exported = {}
            for name, track in tracks.items():
                stem = name.casefold()
                tables = {
                    f"{stem}_events.csv": track.events,
                    f"{stem}_rr_intervals.csv": track.rr_intervals,
                    f"{stem}_rr_hrv_features.csv": track.features,
                }
                for filename, frame in tables.items():
                    path = OUTPUT_DIR / filename
                    frame.to_csv(path, index=False)
                    exported[filename] = {"rows": len(frame), "columns": len(frame.columns)}
            (OUTPUT_DIR / "configuration.json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
            display(pd.DataFrame(exported).T)
            """
        ),
        md(
            """
            ## 8. Automated notebook-level acceptance checks

            These assertions complement—not replace—the package test suite. They
            verify this concrete run: finite input, increasing timestamps, positive
            RR intervals, same-track ownership, defined 100-RR outputs, retained
            reliability flags, and absence of classifier outputs.
            """
        ),
        code(
            """
            for name, track in tracks.items():
                assert np.all(np.diff(track.detector.peak_samples) > 0), name
                assert (track.rr_intervals["rr_ms"] > 0).all(), name
                assert track.features["feature_defined"].any(), name
                assert {"feature_defined", "feature_reliable", "rr_reliability_coverage"}.issubset(track.features.columns)
                assert not track.events["timestamp_moved_after_detection"].any(), name
            assert cascade.metadata["seizure_classifier_applied"] is False
            assert cascade.metadata["artifact_classifier_applied"] is False
            print("PASS — RR/HRV notebook completed all acceptance checks.")
            """
        ),
        md(
            """
            ## Interpretation limits

            - Detector agreement is not expert truth and not signal quality.
            - The 100-RR features are mathematically defined only after 100 intervals.
            - Reliability means the configured detector-support condition passed; it
              is not clinical validity.
            - No target-seizure discrimination, threshold calibration, artifact
              classification, or patient-level validation occurs here.
            - NeuroKit and Zhai remain parallel candidates until expert-annotation
              evidence selects an owner for the intended population and device.
            """
        ),
        *source_appendix([
            "src/ecg_cascade/config.py",
            "src/ecg_cascade/wfdb_io.py",
            "src/ecg_cascade/peaks.py",
            "src/ecg_cascade/zhai.py",
            "src/ecg_cascade/rr_hrv.py",
            "src/ecg_cascade/neurokit_zhai.py",
        ]),
    ]
    return explain_all_code_cells(cells, "rr_hrv")


def morphology_notebook():
    cells = [
        md(
            r"""
            # Standalone ECG morphology preprocessing branch

            **Purpose.** Extract and compare the project's distinct morphology
            measurement families without prematurely merging them into one feature
            matrix.

            This notebook runs, displays, validates, and exports:

            1. fixed published Varon five-QRS / 120-ms eigenvalues;
            2. patient-average symmetric Varon width;
            3. asymmetry-aware symmetric p95 Varon width;
            4. patient-specific asymmetric pre/post-R Varon width;
            5. fixed initial patient/lead whole-cycle templates with correlation,
               residual, derivative-DTW, and previous-beat change;
            6. prominence P–QRS–T landmarks, intervals, amplitudes, area, and polarity.

            Each output is saved separately. The notebook does **not** call three
            Varon eigenvalue sets one final model, and none of the morphology values
            is interpreted as seizure probability or artifact.

            For five captured QRS rows in \(Q\), the fixed Varon core calculates
            \(G=QQ^T\) and returns the five descending eigenvalues of \(G\).
            """
        ),
        *common_setup("morphology", duration_s=120),
        md("## 2. Detect anchors once, then reuse the same track across morphology experiments"),
        code(
            """
            from ecg_cascade import NeuroKitZhaiConfig, run_neurokit_zhai_array

            config = NeuroKitZhaiConfig(compute_inverted_context=False)
            cascade = run_neurokit_zhai_array(ecg, fs, config=config)
            track = cascade.selected.neurokit
            peaks = track.detector.peak_samples
            event_context = track.events
            fixed_varon = track.morphology
            print("NeuroKit anchors:", len(peaks), "Fixed Varon windows:", len(fixed_varon.features))
            """
        ),
        md(
            """
            ## 3. Obtain explicit calibration boundaries and P–QRS–T measurements

            The demo uses the implemented prominence delineator to obtain calibration
            boundaries. This is an **automatic research demonstration**, not manual
            ground truth. In a causal patient-specific study, calibration must come
            only from earlier data and its provenance must be retained.
            """
        ),
        code(
            """
            from ecg_cascade.prominence_morphology import extract_prominence_morphology

            pqrst = extract_prominence_morphology(
                ecg, peaks, fs,
                anchor_track="neurokit",
                patient_id=PATIENT_ID,
                lead_name=CHANNEL,
                segment_start_s=segment.start_s,
            )
            pf = pqrst.features
            calibration = pf.loc[
                pf["complete_qrs"] & pf["physiological_order_valid"]
            ].head(20).copy()
            before_ms = (
                calibration["r_peak_sample_in_segment"].to_numpy(float)
                - pd.to_numeric(calibration["qrs_onset_sample_in_segment"]).to_numpy(float)
            ) * 1000.0 / fs
            after_ms = (
                pd.to_numeric(calibration["qrs_offset_sample_in_segment"]).to_numpy(float)
                - calibration["r_peak_sample_in_segment"].to_numpy(float)
            ) * 1000.0 / fs
            total_ms = before_ms + after_ms
            assert len(total_ms) == 20 and np.all(total_ms > 0)
            display(pd.DataFrame({"onset_to_R_ms": before_ms, "R_to_offset_ms": after_ms, "QRS_total_ms": total_ms}).describe())
            display(pd.Series(pqrst.summary(), name="prominence PQRST summary"))
            """
        ),
        md("## 4. Run every Varon window experiment separately"),
        code(
            """
            from ecg_cascade.morphology import (
                extract_patient_average_varon_morphology,
                extract_patient_asymmetric_varon_morphology,
                extract_patient_required_width_varon_morphology,
            )

            mean_symmetric = extract_patient_average_varon_morphology(
                ecg, peaks, fs, total_ms,
                anchor_track="neurokit", patient_id=PATIENT_ID, lead_name=CHANNEL,
                event_context=event_context,
            )
            p95_symmetric = extract_patient_required_width_varon_morphology(
                ecg, peaks, fs, before_ms, after_ms,
                anchor_track="neurokit", patient_id=PATIENT_ID, lead_name=CHANNEL,
                statistic="p95", event_context=event_context,
            )
            p95_asymmetric = extract_patient_asymmetric_varon_morphology(
                ecg, peaks, fs, before_ms, after_ms,
                anchor_track="neurokit", patient_id=PATIENT_ID, lead_name=CHANNEL,
                statistic="p95", minimum_calibration_beats=5,
                event_context=event_context,
            )
            varon_methods = {
                "fixed_120ms": fixed_varon,
                "patient_mean_symmetric": mean_symmetric,
                "patient_p95_symmetric": p95_symmetric,
                "patient_p95_asymmetric": p95_asymmetric,
            }
            summary_rows = []
            for name, result in varon_methods.items():
                summary_rows.append({
                    "method": name,
                    "pre_R_ms": result.parameters["pre_r_ms_requested"],
                    "post_R_ms": result.parameters["post_r_ms_requested"],
                    "beats": len(result.beat_context),
                    "windows": len(result.features),
                    "defined": int(result.features["published_varon_core_defined"].sum()),
                    "support_pass": int(result.features["support_context_pass"].sum()),
                })
            varon_summary = pd.DataFrame(summary_rows).set_index("method")
            display(varon_summary)
            """
        ),
        code(
            """
            # Visualize the same five-beat window under each capture rule.
            window_index = min(55, len(fixed_varon.features) - 1)
            fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
            for ax, (name, result) in zip(axes.flat, varon_methods.items()):
                row = result.features.iloc[window_index]
                first, last = int(row.start_beat_index), int(row.end_beat_index)
                waveforms = result.beat_waveforms[first:last + 1]
                pre = float(result.parameters["pre_r_ms_realized"])
                post = float(result.parameters["post_r_ms_realized"])
                axis_ms = np.linspace(-pre, post, waveforms.shape[1])
                for beat_number, waveform in enumerate(waveforms, start=1):
                    ax.plot(axis_ms, waveform, lw=1.2, alpha=.78, label=f"beat {beat_number}")
                ax.axvline(0, color="#d38232", ls="--", lw=1)
                ax.set(title=f"{name}\\n{pre:.1f} ms pre-R / {post:.1f} ms post-R", xlabel="Time from R (ms)", ylabel="ECG")
            axes[0, 0].legend(ncol=5, fontsize=7, loc="upper center")
            plt.show()
            """
        ),
        md("## 5. Verify the fixed Varon Gram matrix and eigenvalues directly"),
        code(
            """
            row = fixed_varon.features.iloc[window_index]
            q = fixed_varon.beat_waveforms[int(row.start_beat_index):int(row.end_beat_index) + 1]
            gram = q @ q.T
            direct_eigenvalues = np.linalg.eigvalsh(0.5 * (gram + gram.T))[::-1]
            stored_eigenvalues = row[[f"lambda{i}" for i in range(1, 6)]].to_numpy(float)
            np.testing.assert_allclose(direct_eigenvalues, stored_eigenvalues)

            fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
            image = axes[0].imshow(gram, cmap="magma")
            axes[0].set(title=r"Five-beat Gram matrix $QQ^T$", xlabel="Beat", ylabel="Beat")
            fig.colorbar(image, ax=axes[0], shrink=.8)
            axes[1].bar([f"λ{i}" for i in range(1, 6)], direct_eigenvalues, color="#d48632")
            axes[1].set(title="Descending morphology eigenvalues", ylabel="Waveform energy")
            plt.show()
            display(row[["window_end_time_s", "complete_capture_fraction", "detector_support_fraction", *[f"lambda{i}" for i in range(1, 6)]]])
            """
        ),
        md(
            """
            ## 6. Patient/lead initial-template morphology

            Complete cycles are bounded by neighboring R-R midpoints. Pre-R and
            post-R portions are resampled separately, baseline-centered, and L2
            normalized. The first 20 complete cycles form a fixed template bank;
            later cycles are evaluation rows. Returned values include template
            correlation, normalized RMSE, raw residual RMS/energy, derivative-DTW
            cost, warp fraction, and previous-beat shape change.
            """
        ),
        code(
            """
            from ecg_cascade.patient_template_morphology import extract_patient_template_morphology

            template = extract_patient_template_morphology(
                ecg, peaks, fs,
                anchor_track="neurokit", patient_id=PATIENT_ID, lead_name=CHANNEL,
                calibration_beats=20, segment_start_s=segment.start_s,
            )
            display(pd.Series(template.summary(), name="template morphology summary"))
            template_columns = [
                "anchor_time_s", "calibration_member", "evaluation_eligible",
                "template_correlation", "template_normalized_rmse",
                "template_raw_residual_rms", "template_raw_residual_energy",
                "template_derivative_dtw_cost", "template_dtw_warp_fraction",
                "previous_beat_normalized_rmse", "previous_beat_correlation",
            ]
            display(template.features.loc[template.features["evaluation_eligible"], template_columns].head())
            """
        ),
        code(
            """
            evaluation_index = int(np.flatnonzero(template.features["evaluation_eligible"].to_numpy(bool))[20])
            matched_template = int(template.features.iloc[evaluation_index]["matched_template_index"])
            x = np.linspace(0, 100, template.normalized_waveforms.shape[1])
            fig, axes = plt.subplots(2, 1, figsize=(12, 6), constrained_layout=True)
            axes[0].plot(x, template.normalized_template_bank[matched_template], lw=2.0, label="fixed patient template", color="#2d789e")
            axes[0].plot(x, template.normalized_waveforms[evaluation_index], lw=1.3, label="evaluation beat", color="#c97b2c")
            axes[0].set(title="Normalized patient-template comparison", ylabel="Normalized amplitude")
            axes[0].legend()
            residual = template.normalized_waveforms[evaluation_index] - template.normalized_template_bank[matched_template]
            axes[1].plot(x, residual, color="#7359bf", lw=1.2)
            axes[1].axhline(0, color="#8b999f", lw=.8)
            axes[1].set(title="Normalized residual", xlabel="Resampled cardiac cycle (%)", ylabel="Difference")
            plt.show()
            """
        ),
        md("## 7. Inspect P–QRS–T landmarks and interpretable measurements"),
        code(
            """
            complete_indices = np.flatnonzero((pf["complete_pqrst"] & pf["physiological_order_valid"]).to_numpy(bool))
            beat_index = int(complete_indices[min(15, len(complete_indices) - 1)])
            beat = pf.iloc[beat_index]
            landmark_columns = [
                "p_onset_sample_in_segment", "p_peak_sample_in_segment", "p_offset_sample_in_segment",
                "qrs_onset_sample_in_segment", "r_peak_sample_in_segment", "qrs_offset_sample_in_segment",
                "t_onset_sample_in_segment", "t_peak_sample_in_segment", "t_offset_sample_in_segment",
            ]
            samples = {name: int(beat[name]) for name in landmark_columns}
            first = max(0, samples["p_onset_sample_in_segment"] - int(.08 * fs))
            last = min(ecg.size, samples["t_offset_sample_in_segment"] + int(.08 * fs))
            fig, ax = plt.subplots(figsize=(13, 4))
            local_time = np.arange(first, last) / fs
            ax.plot(local_time, ecg[first:last], color="#244652", lw=1.1)
            landmark_colors = ["#4d8bbc", "#4d8bbc", "#4d8bbc", "#cf8433", "#cf8433", "#cf8433", "#775bc4", "#775bc4", "#775bc4"]
            for (name, sample), color in zip(samples.items(), landmark_colors):
                label = name.replace("_sample_in_segment", "").replace("_", " ")
                ax.scatter(sample / fs, ecg[sample], s=42, color=color, zorder=3)
                ax.annotate(label, (sample / fs, ecg[sample]), xytext=(0, 10), textcoords="offset points", ha="center", fontsize=7, rotation=45)
            ax.set(title="One automatically delineated P–QRS–T complex", xlabel="Time (s)", ylabel="ECG")
            plt.show()
            display(beat[[
                "p_duration_ms", "pr_interval_ms", "qrs_duration_ms", "st_interval_ms", "qt_interval_ms",
                "p_peak_amplitude_from_baseline", "r_anchor_amplitude_from_baseline", "t_peak_amplitude_from_baseline",
                "qrs_peak_to_peak_amplitude", "qrs_signed_area", "qrs_polarity",
                "complete_pqrst", "physiological_order_valid"
            ]])
            """
        ),
        md(
            """
            ## 8. Case study — record 207 exposes morphology, polarity, and coverage

            The all-48-record [morphology fidelity audit](../../docs/17_MORPHOLOGY_FIDELITY_AND_REVIEWER.md)
            identified record 207 as a counterexample hidden by pooled results.
            Its changing bundle-branch morphologies and polarity make waveform
            alignment difficult. In the saved audit, NeuroKit's median absolute
            matched-beat timing error reached 66.7 ms and strict five-beat coverage
            fell to about 0.372, even though pooled NeuroKit timing was much better.

            The [record-207 case report](../../reports/MITBIH_Record_207_Morphology_Polarity_Case_Analysis.pdf)
            also found that original-polarity RR support was only 9.28% in the tested
            architecture, while a dual-polarity context improved coverage—but the
            project correctly refused to select polarity on the same record used to
            demonstrate the gain.

            The successful-run screenshot below is deliberately busy: it is a
            reviewer, not a single-number dashboard. It keeps expert, NeuroKit,
            UNSW/Khamis, Zhai, RR, and five-beat eigenvalue-energy context visible.

            Paper links used by this branch:

            - Varon et al. (2015), five-beat ECG morphology seizure features, [DOI 10.1016/j.jelectrocard.2015.08.020](https://doi.org/10.1016/j.jelectrocard.2015.08.020).
            - Emrich et al. (2024), physiology-informed prominence delineation, [EUSIPCO paper](https://eurasip.org/Proceedings/Eusipco/Eusipco2024/pdfs/0001402.pdf).
            """
        ),
        code(
            """
            from IPython.display import Image

            CASE_SCREENSHOT = PROJECT / "outputs" / "ui_validation" / "record_207_branch_reviewer_full.png"
            CASE_METRICS = PROJECT / "outputs" / "morphology_fidelity_all48_v1" / "record_summary.csv"
            assert CASE_SCREENSHOT.is_file(), CASE_SCREENSHOT
            assert CASE_METRICS.is_file(), CASE_METRICS

            morphology_audit = pd.read_csv(CASE_METRICS)
            morphology_case = morphology_audit.loc[
                morphology_audit["record_id"].astype(str) == "207",
                [
                    "candidate_track", "beat_f1", "median_absolute_timing_error_ms",
                    "five_beat_window_coverage", "median_raw_eigenvalue_relative_l1_error",
                    "median_exploratory_normalized_spectrum_l1_error",
                ],
            ].set_index("candidate_track")
            display(morphology_case.style.format("{:.4f}"))

            assert morphology_case.loc["neurokit", "median_absolute_timing_error_ms"] > 60
            assert morphology_case.loc["neurokit", "five_beat_window_coverage"] < 0.40
            assert morphology_case.loc["zhai", "five_beat_window_coverage"] > 0.90

            display(Image(filename=str(CASE_SCREENSHOT), width=1400))
            print("PASS — record 207 reviewer evidence and quantitative morphology-fidelity rows loaded.")
            """
        ),
        md(
            """
            ### How to read the successful-run screenshot

            1. **Top ECG panel:** black diamonds are expert beats; colored detector
               markers show where each algorithm placed an event. Orange downward
               triangles identify expert beats missed by the active NeuroKit lane.
            2. **Middle morphology panel:** five curves are display-normalized
               lambda energy fractions for consecutive five-beat windows. They help
               compare patterns on one screen; exported raw `lambda1`–`lambda5`
               remain the paper-backed measurements.
            3. **Bottom RR panel:** large alternating RR changes demonstrate how an
               extra or missed ventricular anchor propagates into timing features.
            4. **Correct interpretation:** the screenshot demonstrates a difficult
               morphology/polarity case and a coverage–fidelity tradeoff. It does
               not show a seizure prediction, morphology diagnosis, or automatic
               polarity-selection rule.
            """
        ),
        md("## 9. Export every morphology family separately"),
        code(
            """
            exports = {}
            for name, result in varon_methods.items():
                path = OUTPUT_DIR / f"{name}_varon_features.csv"
                result.features.to_csv(path, index=False)
                exports[path.name] = len(result.features)
            template_path = OUTPUT_DIR / "patient_template_features.csv"
            pqrst_path = OUTPUT_DIR / "prominence_pqrst_features.csv"
            template.features.to_csv(template_path, index=False)
            pf.to_csv(pqrst_path, index=False)
            exports[template_path.name] = len(template.features)
            exports[pqrst_path.name] = len(pf)
            display(pd.Series(exports, name="rows exported").to_frame())
            """
        ),
        md("## 10. Notebook-level acceptance checks and limits"),
        code(
            """
            for name, result in varon_methods.items():
                assert result.features["published_varon_core_defined"].any(), name
                assert all(f"lambda{i}" in result.features for i in range(1, 6)), name
                assert not result.features["seizure_probability_produced"].any(), name
                assert not result.features["artifact_label_produced"].any(), name
            assert template.features["evaluation_eligible"].any()
            assert np.isfinite(template.features.loc[template.features["evaluation_eligible"], "template_correlation"]).all()
            assert pf["complete_qrs"].any()
            assert not pf["seizure_probability_produced"].any()
            print("PASS — all morphology implementations executed and retained separate outputs.")
            """
        ),
        md(
            """
            ### Interpretation limits

            - Automatic prominence boundaries are not manual landmark truth.
            - Personalized calibration in this demo is derived from automatic earlier
              beats; a formal study must use a causal, independently validated split.
            - Eigenvalues are amplitude-sensitive because unreported normalization is
              not added to the published Varon core.
            - Template novelty can reflect physiology, ectopy, lead movement, gain,
              artifact, or landmark error. No cause is assigned.
            - The final matrix should include only the selected Varon experiment after
              held-out comparison, not three nearly duplicated eigenvalue sets.
            """
        ),
        *source_appendix([
            "src/ecg_cascade/morphology.py",
            "src/ecg_cascade/patient_template_morphology.py",
            "src/ecg_cascade/prominence_morphology.py",
            "src/ecg_cascade/delineation_validation.py",
            "src/ecg_cascade/neurokit_zhai.py",
        ]),
    ]
    return explain_all_code_cells(cells, "morphology")


def signal_quality_notebook():
    cells = [
        md(
            r"""
            # Standalone signal-quality / artifact-context preprocessing branch

            **Correct name and scope.** The implemented branch returns continuous
            signal-quality indices, per-feature availability booleans, and explicit
            reasons when prerequisites are missing. It is **not yet an artifact
            classifier** and does not emit a universal clean/artifact decision.

            The notebook covers finite samples, saturation, flatline, ADC rails,
            amplitude, baseline/QRS-band power, skewness/kurtosis, high-frequency
            noise, bSQI, qSQI, Ho RR support, cross-lead iSQI, beat-template
            correlation, Galeotti baseline/mains/residual features, Menon Fourier
            score, trailing causal windows, and controlled degradations.

            It demonstrates how measurements respond to synthetic corruption but
            intentionally does not turn that response into a trained threshold.
            """
        ),
        *common_setup("signal_quality", duration_s=60),
        md(
            """
            ## 2. Unit declaration and prerequisite signals

            MIT–BIH physical signals are stored in mV, so amplitude-valued SQIs are
            converted with `uv_per_input_unit=1000`. The value must be changed for a
            source stored in volts, µV, or ADC counts. The second `V5` channel is
            loaded only to demonstrate cross-lead agreement; it is not averaged with
            `MLII`.
            """
        ),
        code(
            """
            from ecg_cascade import NeuroKitZhaiConfig, run_neurokit_zhai_array
            from ecg_cascade.peaks import detect_r_peaks
            from ecg_cascade.prominence_morphology import extract_prominence_morphology

            header = wfdb.rdheader(str(RECORD_PATH))
            channel_index = header.sig_name.index(CHANNEL)
            print("WFDB header units:", dict(zip(header.sig_name, header.units)))
            assert header.units[channel_index].casefold() == "mv"
            UV_PER_INPUT_UNIT = 1000.0
            MAINS_FREQUENCY_HZ = 60.0  # explicit metadata assumption for MIT–BIH demo

            second = load_wfdb_segment(RECORD_PATH, channel="V5", start_s=START_S, duration_s=DURATION_S)
            assert second.sampling_rate_hz == fs and second.samples.size == ecg.size

            cascade = run_neurokit_zhai_array(
                ecg, fs, config=NeuroKitZhaiConfig(compute_inverted_context=False)
            )
            primary_peaks = cascade.selected.neurokit.detector.peak_samples
            secondary_peaks = cascade.selected.zhai.detector.peak_samples
            other_lead_peaks = detect_r_peaks(second.samples, fs, method="neurokit").peak_samples
            pqrst = extract_prominence_morphology(
                ecg, primary_peaks, fs,
                anchor_track="neurokit", patient_id=PATIENT_ID, lead_name=CHANNEL,
            )
            qrs_onsets = pqrst.landmarks["qrs_onset"]
            qrs_onsets = np.rint(qrs_onsets[np.isfinite(qrs_onsets)]).astype(np.int64)
            print({"primary_peaks": len(primary_peaks), "secondary_peaks": len(secondary_peaks),
                   "other_lead_peaks": len(other_lead_peaks), "qrs_onsets": len(qrs_onsets)})
            """
        ),
        md("## 3. Calculate the complete paper-named feature vector for one 10-second window"),
        code(
            """
            from ecg_cascade.signal_quality import extract_signal_quality_window

            first = int(20 * fs)
            last = int(30 * fs)
            window = ecg[first:last]

            def local_events(events, start, stop):
                events = np.asarray(events, dtype=np.int64)
                keep = (events >= start) & (events < stop)
                return events[keep] - start

            clean_quality = extract_signal_quality_window(
                window, fs,
                uv_per_input_unit=UV_PER_INPUT_UNIT,
                primary_peak_samples=local_events(primary_peaks, first, last),
                secondary_peak_samples=local_events(secondary_peaks, first, last),
                qrs_onset_samples=local_events(qrs_onsets, first, last),
                other_lead_peak_samples=[local_events(other_lead_peaks, first, last)],
                mains_frequency_hz=MAINS_FREQUENCY_HZ,
                flat_tolerance_input_units=0.0,
            )
            quality_table = pd.DataFrame({
                "value": clean_quality.values,
                "available": clean_quality.available,
                "reason": clean_quality.reasons,
            })
            display(quality_table)
            print("Available:", int(quality_table.available.sum()), "/", len(quality_table))
            """
        ),
        md(
            r"""
            ## 4. Meaning of the major SQI families

            - **Basic integrity:** finite fraction, flat runs, saturation, rails.
            - **Amplitude:** absolute maximum, peak-to-peak, RMS in µV.
            - **Spectral:** \(basSQI=1-P_{0-1}/P_{0-40}\),
              \(pSQI=P_{5-15}/P_{5-40}\), and Li's QRS power ratio.
            - **Statistics:** population skewness and Pearson kurtosis.
            - **Detector agreement:** Li bSQI Jaccard at 150 ms, Zhao/Kristof
              qSQI Dice at 75 ms, and Ho endpoint RR support at 150 ms. These
              equations are intentionally distinct.
            - **Cross-lead context:** maximum same-detector Jaccard agreement.
            - **Waveform consistency:** Orphanidou template correlation and Menon
              Fourier score.
            - **Noise decomposition:** explicit-metadata Galeotti baseline and mains
              RMS; residual RMS requires externally supplied beat groups.

            Missing prerequisites remain visible in the `available` and `reason`
            columns rather than being silently imputed.
            """
        ),
        md("## 5. Controlled degradations — diagnostic response, not classification"),
        code(
            """
            from ecg_cascade.peaks import detect_neurokit_gradient
            from ecg_cascade.zhai import detect_zhai_template

            local_t = np.arange(window.size) / fs
            rng = np.random.default_rng(20260820)
            variants = {
                "clean": window.copy(),
                "baseline_wander": window + 0.60 * np.sin(2 * np.pi * 0.30 * local_t),
                "high_frequency": window + 0.10 * np.sin(2 * np.pi * 45.0 * local_t) + 0.025 * rng.normal(size=window.size),
                "flatline_2s": window.copy(),
                "amplitude_clipped": np.clip(window, -0.45, 0.45),
            }
            variants["flatline_2s"][int(4 * fs):int(6 * fs)] = variants["flatline_2s"][int(4 * fs)]

            def detector_pair(signal):
                nk_run = detect_neurokit_gradient(
                    signal, fs, orientation="original",
                    minimum_delay_ms=300.0, minimum_delay_inclusive=False,
                )
                zhai_run = detect_zhai_template(signal, fs, orientation="original")
                return nk_run.peak_samples, zhai_run.detector.peak_samples

            comparison_records = []
            comparison_results = {}
            for name, signal in variants.items():
                p1, p2 = detector_pair(signal)
                result = extract_signal_quality_window(
                    signal, fs,
                    uv_per_input_unit=UV_PER_INPUT_UNIT,
                    primary_peak_samples=p1,
                    secondary_peak_samples=p2,
                    mains_frequency_hz=MAINS_FREQUENCY_HZ,
                    flat_tolerance_input_units=0.0,
                )
                comparison_results[name] = result
                comparison_records.append({"condition": name, **result.values})
            degradation = pd.DataFrame(comparison_records).set_index("condition")
            selected_metrics = [
                "finite_fraction", "peak_to_peak_uv", "flat_fraction", "bassqi_clifford",
                "psqi_clifford", "hf_rms_uv", "pearson_kurtosis",
                "bsqi_li2008_jaccard_150ms", "qsqi_zhao2018_dice_75ms",
                "rr_support_ho2024_150ms", "template_corr_orphanidou",
            ]
            display(degradation[selected_metrics])
            """
        ),
        code(
            """
            fig, axes = plt.subplots(len(variants), 1, figsize=(13, 10), sharex=True, constrained_layout=True)
            for ax, (name, signal) in zip(axes, variants.items()):
                ax.plot(local_t, signal, lw=.8, color="#245668")
                ax.set_ylabel(name.replace("_", " "), rotation=0, ha="right", va="center")
            axes[0].set_title("Controlled signal degradations used for the SQI diagnostic")
            axes[-1].set_xlabel("Time in window (s)")
            plt.show()

            matrix = degradation[selected_metrics].copy()
            normalized = matrix.copy()
            for column in matrix:
                values = matrix[column].to_numpy(float)
                finite = np.isfinite(values)
                if finite.any() and np.ptp(values[finite]) > 0:
                    normalized.loc[finite, column] = (values[finite] - values[finite].min()) / np.ptp(values[finite])
                else:
                    normalized.loc[finite, column] = .5
            fig, ax = plt.subplots(figsize=(13, 4.2))
            image = ax.imshow(normalized.to_numpy(float), aspect="auto", cmap="viridis", vmin=0, vmax=1)
            ax.set_xticks(range(len(selected_metrics)), [name.replace("_", "\\n") for name in selected_metrics], rotation=45, ha="right", fontsize=7)
            ax.set_yticks(range(len(normalized)), normalized.index)
            ax.set_title("Per-column relative SQI response (not an artifact score)")
            fig.colorbar(image, ax=ax, label="relative within this diagnostic")
            plt.show()
            """
        ),
        md(
            """
            ## 6. Case study — controlled NSTDB electrode-motion noise

            The prior [signal-quality implementation audit](../../docs/21_SIGNAL_QUALITY_BRANCH_V0.md)
            evaluated 12 electrode-motion windows from NSTDB records 118/119 at
            +24 through −6 dB. Because noise is added at known SNR levels, this is a
            useful monotonic-response case. It is still **not** a manually labelled
            artifact-classification benchmark.

            The earlier run found strong positive rank correlation with improving
            SNR for basSQI, qSQI, and bSQI, while amplitude RMS and high-frequency
            RMS increased as SNR worsened. Menon's Fourier score was nearly
            non-monotonic, which is exactly why a multi-feature context vector is
            retained instead of a universal hard veto.

            The conceptual evidence review, [*Signal Quality as Context in ECG
            Diagnostic Fusion*](../../reports/Signal_Quality_as_Context_Evidence_Review.pdf),
            concluded that quality supplied beside physiology has direct precedent,
            but this exact quality-plus-ECG seizure formulation remains unvalidated.
            The case cell therefore visualizes response and never reports accuracy,
            sensitivity, specificity, or an artifact label.

            Useful method precedents include Zhao and Zhang's detector-agreement
            SQI ([DOI 10.3389/fphys.2018.00727](https://doi.org/10.3389/fphys.2018.00727))
            and Li, Mark, and Clifford's bSQI framework
            ([open article](https://pmc.ncbi.nlm.nih.gov/articles/PMC2259026/)).
            """
        ),
        code(
            """
            NSTDB_CASE = PROJECT / "outputs" / "signal_quality_nstdb_v0"
            feature_path = NSTDB_CASE / "window_features.csv"
            rank_path = NSTDB_CASE / "snr_rank_diagnostics.csv"
            assert feature_path.is_file(), feature_path
            assert rank_path.is_file(), rank_path

            nstdb_features = pd.read_csv(feature_path)
            nstdb_rank = pd.read_csv(rank_path).set_index("feature")
            nstdb_noisy = nstdb_features.loc[
                nstdb_features["condition"].eq("electrode_motion")
            ].copy()
            nstdb_noisy["snr_db"] = pd.to_numeric(nstdb_noisy["snr_db"], errors="raise")
            nstdb_noisy = nstdb_noisy.sort_values(["source_record", "snr_db"])

            display(nstdb_rank.sort_values("spearman_with_snr_db", ascending=False))
            assert len(nstdb_noisy) == 12
            assert nstdb_rank.loc["bassqi_clifford", "spearman_with_snr_db"] > 0.95
            assert nstdb_rank.loc["rms_uv", "spearman_with_snr_db"] < -0.95

            trajectory_metrics = [
                ("bassqi_clifford", "basSQI", "higher with improving SNR"),
                ("qsqi_zhao2018_dice_75ms", "qSQI", "higher with improving SNR"),
                ("rms_uv", "RMS (µV)", "larger as noise worsens"),
            ]
            fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
            for ax, (column, label, interpretation) in zip(axes, trajectory_metrics):
                for source_record, group in nstdb_noisy.groupby("source_record"):
                    ax.plot(group["snr_db"], group[column], marker="o", lw=1.5, label=f"record {source_record}")
                ax.set(title=f"{label}\\n{interpretation}", xlabel="Added-noise SNR (dB)", ylabel=label)
            axes[0].legend()
            fig.suptitle("NSTDB electrode-motion case: feature trajectories are responses, not labels", fontweight="bold")
            plt.show()

            rank_plot = nstdb_rank.sort_values("spearman_with_snr_db")
            colors = np.where(rank_plot["spearman_with_snr_db"] >= 0, "#188a7a", "#d17a35")
            fig, ax = plt.subplots(figsize=(11, 5.2))
            ax.barh(rank_plot.index, rank_plot["spearman_with_snr_db"], color=colors)
            ax.axvline(0, color="#56666d", lw=1)
            ax.set(
                title="Spearman response to improving SNR across 12 noisy windows",
                xlabel="rho with SNR (response direction, not classifier performance)",
            )
            plt.show()
            print("PASS — NSTDB case evidence loaded: 12 electrode-motion windows, no label fitted.")
            """
        ),
        md(
            """
            ### How to read the successful-run screenshots

            - Moving right on the three trajectory panels means **less severe added
              noise**. basSQI and qSQI generally rise; RMS falls. The two record
              traces need not overlap because underlying ECG amplitude/morphology
              differ.
            - In the horizontal bar chart, positive rho means a feature tends to
              increase with improving SNR; negative rho means it tends to decrease.
              Direction is not "good" versus "bad."
            - A correlation near zero, such as the recorded Menon result, means this
              particular feature did not rank these 12 windows monotonically. It
              should remain a comparator, not be forced into a hard threshold.
            - Correct execution is the expected table, two figures, 12-window
              assertion, and final PASS sentence—not a clean/artifact prediction.
            """
        ),
        md("## 7. Causal trailing 10-second quality windows across the recording"),
        code(
            """
            from ecg_cascade.signal_quality import extract_trailing_signal_quality_windows

            quality_series = extract_trailing_signal_quality_windows(
                ecg, fs,
                uv_per_input_unit=UV_PER_INPUT_UNIT,
                window_s=10.0,
                step_s=5.0,
                primary_peak_samples=primary_peaks,
                secondary_peak_samples=secondary_peaks,
                qrs_onset_samples=qrs_onsets,
                other_lead_peak_samples=[other_lead_peaks],
                mains_frequency_hz=MAINS_FREQUENCY_HZ,
            )
            trailing = quality_series.to_frame()
            trailing["window_end_s"] = trailing["window_end_sample_exclusive"] / fs
            display(trailing.head())

            fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True, constrained_layout=True)
            plotted = [
                ("bassqi_clifford", "basSQI"),
                ("qsqi_zhao2018_dice_75ms", "qSQI detector Dice"),
                ("template_corr_orphanidou", "template correlation"),
            ]
            for ax, (column, label) in zip(axes, plotted):
                ax.plot(trailing["window_end_s"], trailing[column], marker="o", ms=3, lw=1.2)
                ax.set_ylabel(label)
            axes[0].set_title("Trailing causal signal-quality measurements")
            axes[-1].set_xlabel("Exclusive window end time (s)")
            plt.show()
            """
        ),
        md("## 8. Export measurements, availability, reasons, and configuration"),
        code(
            """
            quality_table.to_csv(OUTPUT_DIR / "single_window_quality_with_availability.csv")
            degradation.to_csv(OUTPUT_DIR / "controlled_degradation_quality.csv")
            trailing.to_csv(OUTPUT_DIR / "trailing_signal_quality_windows.csv", index=False)
            (OUTPUT_DIR / "single_window_parameters.json").write_text(
                json.dumps(clean_quality.parameters, indent=2), encoding="utf-8"
            )
            print("Exported files:")
            for path in sorted(OUTPUT_DIR.glob("*")):
                print(" -", path.name, path.stat().st_size, "bytes")
            """
        ),
        md("## 9. Notebook-level acceptance checks and limits"),
        code(
            """
            assert clean_quality.available["finite_fraction"]
            assert clean_quality.available["bsqi_li2008_jaccard_150ms"]
            assert clean_quality.available["qsqi_zhao2018_dice_75ms"]
            assert clean_quality.available["isqi_li2008_max_jaccard_150ms"]
            assert comparison_results["flatline_2s"].values["flat_fraction"] > clean_quality.values["flat_fraction"]
            assert len(trailing) == len(quality_series.windows) > 0
            forbidden = {"artifact_label", "artifact_probability", "clean_label", "seizure_probability"}
            assert not forbidden.intersection(clean_quality.values)
            print("PASS — signal-quality measurements executed without inventing an artifact classifier.")
            """
        ),
        md(
            """
            ### Interpretation limits

            - Synthetic degradation response is not artifact-classifier accuracy.
            - Device-specific rails and high-frequency thresholds remain unavailable
              unless the caller supplies them.
            - SQI thresholds cannot be transferred from one dataset/device without
              patient-wise calibration and manually labelled artifact windows.
            - Detector agreement can fail on clean pathological rhythms and can agree
              on a shared error; it is context, not truth.
            - BUT-QDB or equivalent labelled data remains necessary before fitting and
              evaluating an artifact classifier.
            """
        ),
        *source_appendix([
            "src/ecg_cascade/signal_quality.py",
            "src/ecg_cascade/rr_reliability_context.py",
            "src/ecg_cascade/peaks.py",
            "src/ecg_cascade/zhai.py",
            "scripts/validate_signal_quality_nstdb.py",
        ]),
    ]
    return explain_all_code_cells(cells, "signal_quality")


def conduction_notebook():
    cells = [
        md(
            r"""
            # Standalone conduction and repolarization preprocessing branch

            **Purpose.** Convert explicit R anchors and aligned P/QRS/T landmarks
            into beatwise conduction/repolarization intervals, QT-correction
            alternatives, first differences, causal 30-beat variability summaries,
            five-beat information windows, and nine-beat QT context.

            ### Core beat measurements

            \[
            PR=P_{onset}\rightarrow QRS_{onset},\quad
            QRS=QRS_{onset}\rightarrow QRS_{offset},\quad
            QT=QRS_{onset}\rightarrow T_{offset},\quad
            JT=QRS_{offset}\rightarrow T_{offset}
            \]

            \[
            QTc_B=\frac{QT}{\sqrt{RR_s}},\qquad
            QTc_F=\frac{QT}{\sqrt[3]{RR_s}},\qquad
            QTc_{Framingham}=QT+154(1-RR_s)
            \]

            The branch is information-only: it does not diagnose conduction disease,
            apply clinical thresholds, label abnormality, or predict seizures.
            """
        ),
        *common_setup("conduction", duration_s=120),
        md("## 2. Detect R anchors and delineate P–QRS–T landmarks"),
        code(
            """
            from ecg_cascade import NeuroKitZhaiConfig, run_neurokit_zhai_array
            from ecg_cascade.prominence_morphology import extract_prominence_morphology

            cascade = run_neurokit_zhai_array(
                ecg, fs, config=NeuroKitZhaiConfig(compute_inverted_context=False)
            )
            track = cascade.selected.neurokit
            prominence = extract_prominence_morphology(
                ecg, track.detector.peak_samples, fs,
                anchor_track="neurokit", patient_id=PATIENT_ID, lead_name=CHANNEL,
                segment_start_s=segment.start_s,
            )
            pqrst = prominence.features
            display(pd.Series(prominence.summary(), name="delineation summary"))
            display(pqrst.head(4))
            """
        ),
        md(
            """
            The prominence delineator is an upstream measurement method, not manual
            reference truth. Missing landmarks remain missing. `complete_pqrst` and
            `physiological_order_valid` make per-beat availability explicit.
            """
        ),
        code(
            """
            valid = np.flatnonzero((pqrst["complete_pqrst"] & pqrst["physiological_order_valid"]).to_numpy(bool))
            beat_index = int(valid[min(18, len(valid) - 1)])
            beat = pqrst.iloc[beat_index]
            columns = [
                "p_onset_sample_in_segment", "p_peak_sample_in_segment", "p_offset_sample_in_segment",
                "qrs_onset_sample_in_segment", "r_peak_sample_in_segment", "qrs_offset_sample_in_segment",
                "t_onset_sample_in_segment", "t_peak_sample_in_segment", "t_offset_sample_in_segment",
            ]
            landmarks = {name: int(beat[name]) for name in columns}
            first = max(0, landmarks["p_onset_sample_in_segment"] - int(.08 * fs))
            last = min(ecg.size, landmarks["t_offset_sample_in_segment"] + int(.08 * fs))
            fig, ax = plt.subplots(figsize=(13, 4))
            ax.plot(np.arange(first, last) / fs, ecg[first:last], color="#234b5a", lw=1.1)
            for name, sample in landmarks.items():
                short = name.replace("_sample_in_segment", "").replace("_", " ")
                ax.scatter(sample / fs, ecg[sample], s=38, zorder=3)
                ax.annotate(short, (sample / fs, ecg[sample]), xytext=(0, 10), textcoords="offset points", ha="center", fontsize=7, rotation=45)
            ax.set(title="Landmarks consumed by the conduction branch", xlabel="Time (s)", ylabel="ECG")
            plt.show()
            """
        ),
        md("## 3. Run frozen beatwise timing equations and 30-beat summaries"),
        code(
            """
            from ecg_cascade.conduction_timing import extract_conduction_timing

            timing = extract_conduction_timing(
                pqrst, fs,
                variability_window_beats=30,
                minimum_valid_fraction=0.8,
                landmark_method=prominence.parameters["method"],
            )
            beats = timing.beat_features
            rolling30 = timing.rolling_features
            display(pd.Series(timing.summary(), name="conduction timing v1 summary"))
            display(beats[[
                "beat_index", "r_peak_time_s", "preceding_rr_ms", "instantaneous_heart_rate_bpm",
                "pr_interval_ms", "qrs_duration_ms", "qt_interval_ms", "jt_interval_ms",
                "qtc_bazett_ms", "qtc_fridericia_ms", "qtc_framingham_ms",
                "all_raw_wave_intervals_defined"
            ]].head(8))
            """
        ),
        md("## 4. Direct equation checks for one complete beat"),
        code(
            """
            check_index = int(np.flatnonzero(beats["all_raw_wave_intervals_defined"].to_numpy(bool))[5])
            raw = pqrst.iloc[check_index]
            measured = beats.iloc[check_index]
            direct = {
                "pr_interval_ms": (float(raw.qrs_onset_sample_in_segment) - float(raw.p_onset_sample_in_segment)) * 1000 / fs,
                "qrs_duration_ms": (float(raw.qrs_offset_sample_in_segment) - float(raw.qrs_onset_sample_in_segment)) * 1000 / fs,
                "qt_interval_ms": (float(raw.t_offset_sample_in_segment) - float(raw.qrs_onset_sample_in_segment)) * 1000 / fs,
                "jt_interval_ms": (float(raw.t_offset_sample_in_segment) - float(raw.qrs_offset_sample_in_segment)) * 1000 / fs,
            }
            for column, value in direct.items():
                np.testing.assert_allclose(value, float(measured[column]))
            rr_s = float(measured.preceding_rr_ms) / 1000.0
            np.testing.assert_allclose(float(measured.qtc_bazett_ms), direct["qt_interval_ms"] / np.sqrt(rr_s))
            np.testing.assert_allclose(float(measured.qtc_fridericia_ms), direct["qt_interval_ms"] / np.cbrt(rr_s))
            display(pd.DataFrame({"direct_calculation": direct, "stored_value": {key: measured[key] for key in direct}}))
            print("Beatwise interval and QTc equation checks passed.")
            """
        ),
        code(
            """
            fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True, constrained_layout=True)
            for column, label, color in [
                ("pr_interval_ms", "PR", "#3d83a7"),
                ("qrs_duration_ms", "QRS", "#d38232"),
                ("qt_interval_ms", "QT", "#7659c4"),
                ("jt_interval_ms", "JT", "#9b72b4"),
            ]:
                axes[0].plot(beats["r_peak_time_s"], beats[column], label=label, lw=1.1, alpha=.85, color=color)
            axes[0].set(title="Beatwise raw intervals", ylabel="Duration (ms)")
            axes[0].legend(ncol=4)
            axes[1].plot(beats["r_peak_time_s"], beats["qtc_bazett_ms"], label="Bazett", lw=1)
            axes[1].plot(beats["r_peak_time_s"], beats["qtc_fridericia_ms"], label="Fridericia", lw=1)
            axes[1].plot(beats["r_peak_time_s"], beats["qtc_framingham_ms"], label="Framingham", lw=1)
            axes[1].set(title="Named QT-correction alternatives", ylabel="QTc (ms)")
            axes[1].legend(ncol=3)
            defined = rolling30["qt_interval_variability30_defined"]
            axes[2].plot(rolling30.loc[defined, "r_peak_time_s"], rolling30.loc[defined, "qt_interval_sd30_ms"], label="QT SD30")
            axes[2].plot(rolling30.loc[defined, "r_peak_time_s"], rolling30.loc[defined, "qt_interval_rmssd30_ms"], label="QT RMSSD30")
            axes[2].set(title="Causal 30-beat QT variability summaries", xlabel="Time (s)", ylabel="ms")
            axes[2].legend()
            plt.show()
            """
        ),
        md(
            """
            ## 5. Run the information-only five-beat and nine-beat views

            The v2 branch reuses the frozen beat equations, then returns long-format
            trailing five-beat descriptive summaries and a separate nine-beat QT
            context. These values are deliberately marked `final_feature_matrix_eligible=False`
            until interval validation and target-population testing are complete.
            """
        ),
        code(
            """
            from ecg_cascade.conduction_information import extract_conduction_information

            information = extract_conduction_information(
                pqrst, fs,
                window_beats=5,
                minimum_valid_beats=4,
                eligibility_column=None,
                landmark_method=prominence.parameters["method"],
            )
            display(pd.Series(information.summary(), name="conduction information v2 summary"))
            display(information.information_windows.head(12))
            display(information.qt_nine_beat_context.dropna(how="all", axis=1).tail(5))
            """
        ),
        md(
            """
            ## 6. Case study — record 231 separates atrial activity from ventricular RR

            The prior [record-231 conduction case report](../../reports/MITBIH_Record_231_Conduction_False_Detection_Case_Analysis.pdf)
            documents 2:1/Mobitz-II conduction context. Smaller deflections occurred
            between large annotated QRS complexes. Their identity as non-conducted
            atrial activity is physiologically plausible, but the MIT–BIH beat file
            does not directly label every atrial deflection, so the report correctly
            preserves that statement as an inference.

            This is precisely why a conduction branch exists. An intervening atrial
            wave may be real physiology yet must not become a ventricular timestamp
            and create false short RR intervals. The cell below reruns the current
            NeuroKit–Zhai branch on the report's 461.8–470.3 s strip and shows which
            events align with expert ventricular beats.

            Paper evidence constraining this branch:

            - Diab et al. (2025) retained ordered raw P/Q/S/T timing dynamics rather
              than inventing five-beat interval averages, [DOI 10.1016/j.neucli.2025.103098](https://doi.org/10.1016/j.neucli.2025.103098).
            - Brotherstone et al. (2010) used nine consecutive QT/RR pairs and named
              QT corrections, [DOI 10.1111/j.1528-1167.2009.02281.x](https://doi.org/10.1111/j.1528-1167.2009.02281.x).
            - The EHRA/ESC position statement warns that QT variability is easily
              contaminated by T-end error and morphology instability,
              [DOI 10.1093/europace/euv405](https://doi.org/10.1093/europace/euv405).
            """
        ),
        code(
            """
            from ecg_cascade.wfdb_io import load_wfdb_beat_annotations

            CASE_RECORD = (
                ROOT / "Datasets" / "mit-bih-arrhythmia-database-1.0"
                / "mit-bih-arrhythmia-database-1.0.0" / "231"
            )
            case_segment = load_wfdb_segment(
                CASE_RECORD, channel="MLII", start_s=461.8, duration_s=8.5
            )
            case_cascade = run_neurokit_zhai_array(
                case_segment.samples,
                case_segment.sampling_rate_hz,
                segment_start_s=case_segment.start_s,
                config=NeuroKitZhaiConfig(compute_inverted_context=False),
                source_metadata={"record": "231", "case": "intervening deflections"},
            )

            case_reference_samples, case_reference_symbols = load_wfdb_beat_annotations(CASE_RECORD)
            case_reference_times_all = case_reference_samples / case_segment.sampling_rate_hz
            case_reference_keep = (
                (case_reference_times_all >= case_segment.start_s)
                & (case_reference_times_all < case_segment.start_s + case_segment.duration_s)
            )
            case_reference_times = case_reference_times_all[case_reference_keep]
            case_reference_symbols = np.asarray(case_reference_symbols)[case_reference_keep]
            case_nk_times = (
                case_segment.start_s
                + case_cascade.selected.neurokit.detector.peak_samples / case_segment.sampling_rate_hz
            )
            case_zhai_times = (
                case_segment.start_s
                + case_cascade.selected.zhai.detector.peak_samples / case_segment.sampling_rate_hz
            )
            case_counts = pd.Series({
                "expert_ventricular_beats": case_reference_times.size,
                "NeuroKit_events": case_nk_times.size,
                "Zhai_events": case_zhai_times.size,
            }, name="count")
            display(case_counts.to_frame())

            assert case_reference_times.size == 5
            assert case_nk_times.size >= case_reference_times.size + 4
            assert abs(case_zhai_times.size - case_reference_times.size) <= 1

            case_time = case_segment.start_s + np.arange(case_segment.samples.size) / case_segment.sampling_rate_hz
            fig, ax = plt.subplots(figsize=(14, 4.8))
            ax.plot(case_time, case_segment.samples, color="#244652", lw=1.0, label="raw MLII ECG")
            ax.scatter(
                case_reference_times,
                np.interp(case_reference_times, case_time, case_segment.samples),
                marker="D", facecolors="none", edgecolors="#111111", s=58, linewidths=1.3,
                label="expert .atr ventricular beat",
            )
            ax.scatter(
                case_nk_times,
                np.interp(case_nk_times, case_time, case_segment.samples),
                marker="x", color="#d94f3d", s=48, linewidths=1.7,
                label="NeuroKit event",
            )
            ax.scatter(
                case_zhai_times,
                np.interp(case_zhai_times, case_time, case_segment.samples),
                marker="|", color="#7057b8", s=120, linewidths=2.2,
                label="Zhai event",
            )
            ax.set(
                title="Case study: MIT–BIH record 231 — intervening deflections are not ventricular beats",
                xlabel="Time from recording start (s)", ylabel="MLII (mV)",
            )
            ax.legend(ncol=4, loc="upper center")
            plt.show()
            print("PASS — record 231 reproduces the conduction-related extra-event pattern.")
            """
        ),
        md(
            """
            ### How to read the successful-run screenshot

            1. Hollow black diamonds mark the five expert ventricular beats on the
               large QRS complexes.
            2. Purple Zhai markers should align with those large complexes in this
               strip. Red NeuroKit crosses also appear on smaller intervening
               deflections, producing more events than expert ventricular beats.
            3. The screenshot supports the narrow statement that this detector can
               confuse intervening physiology with ventricular events here. It does
               not prove the exact atrial-wave identity, declare artifact, diagnose
               heart block, or establish universal detector superiority.
            4. The conduction extractor must therefore consume a validated
               ventricular anchor lane and retain P/QRS/T landmark availability.
            """
        ),
        md("## 7. Export every conduction table and its audit metadata"),
        code(
            """
            tables = {
                "prominence_pqrst_features.csv": pqrst,
                "conduction_beat_features.csv": beats,
                "conduction_rolling30_features.csv": rolling30,
                "conduction_information_windows_long.csv": information.information_windows,
                "conduction_qt_nine_beat_context.csv": information.qt_nine_beat_context,
            }
            for filename, frame in tables.items():
                frame.to_csv(OUTPUT_DIR / filename, index=False)
            metadata = {
                "delineation": prominence.parameters,
                "timing": timing.parameters,
                "timing_diagnostics": timing.diagnostics,
                "information": information.parameters,
                "information_diagnostics": information.diagnostics,
            }
            (OUTPUT_DIR / "conduction_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            display(pd.DataFrame({name: {"rows": len(frame), "columns": len(frame.columns)} for name, frame in tables.items()}).T)
            """
        ),
        md("## 8. Notebook-level acceptance checks and limits"),
        code(
            """
            assert len(beats) == len(pqrst) == len(track.detector.peak_samples)
            assert beats["qrs_duration_defined"].any()
            assert beats["qt_interval_defined"].any()
            assert rolling30["qt_interval_variability30_defined"].any()
            assert information.information_windows[f"history{information.parameters['window_beats']}_complete"].any()
            assert not beats["seizure_probability_produced"].any()
            assert not beats["clinical_interpretation_produced"].any()
            assert not information.beat_measurements["final_feature_matrix_eligible"].any()
            assert not information.beat_measurements["abnormality_label_produced"].any()
            print("PASS — conduction timing and information branches executed with all safety assertions intact.")
            """
        ),
        md(
            """
            ### Interpretation limits

            - P-onset and T-offset error tails can be as large as, or larger than,
              the physiological variability being measured.
            - QT correction changes rate dependence; it does not create a diagnosis.
            - Thirty-beat QTVI/STV summaries are exploratory and are not equivalent to
              studies using longer stationary recordings.
            - Five-beat descriptive changes can reflect rate, ectopy, physiology,
              artifact, lead movement, or landmark error. No cause is assigned.
            - External signal-quality eligibility can be supplied later, but this
              branch never invents or reinterprets an artifact score.
            """
        ),
        *source_appendix([
            "src/ecg_cascade/prominence_morphology.py",
            "src/ecg_cascade/conduction_timing.py",
            "src/ecg_cascade/conduction_information.py",
            "src/ecg_cascade/delineation_validation.py",
            "scripts/run_conduction_timing.py",
            "scripts/run_conduction_information.py",
        ]),
    ]
    return explain_all_code_cells(cells, "conduction")


def write_notebook(filename: str, cells: list) -> Path:
    notebook = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3 (ECG cascade)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
            "ecg_cascade": {
                "generated_by": Path(__file__).name,
                "purpose": "standalone_detailed_branch_reproduction",
                "clinical_use": False,
            },
        },
    )
    path = NOTEBOOK_DIR / filename
    nbf.write(notebook, path)
    return path


def main() -> int:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    outputs = [
        write_notebook("01_RR_HRV_Standalone_Detailed.ipynb", rr_hrv_notebook()),
        write_notebook("02_Morphology_Standalone_Detailed.ipynb", morphology_notebook()),
        write_notebook("03_Signal_Quality_Artifact_Context_Standalone_Detailed.ipynb", signal_quality_notebook()),
        write_notebook("04_Conduction_Standalone_Detailed.ipynb", conduction_notebook()),
    ]
    print("Generated notebooks:")
    for path in outputs:
        print(f" - {path.relative_to(ROOT)} ({path.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
