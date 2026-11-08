# Reproducibility and evidence

The quickstart reproduces software behavior on artificial data. It does not reproduce original strategy performance or the academic extension. The generator uses NumPy's `default_rng` with seed 1729 and five fixed complete winter sessions. CSV values are rounded to ten decimal places before baseline ingestion. The tested environment is recorded separately; numerical output on another dependency stack may differ at rounding boundaries.

The baseline demonstration executes the actual extracted feature and signal logic, not a monkeypatched strategy. Focused execution tests separately inject artificial features to isolate exits, next-bar timing, costs, session closure and actual initial-stop risk. These two kinds of evidence answer different questions.

Intake's demo uses a fixed clock and `httpx.MockTransport`. Its run IDs include fresh random identifiers, so raw catalogs/manifests are not promised byte-identical across executions. The printed summary excludes those volatile IDs. Fixtures are synthetic; actual arXiv retrieval was not tested.

The candidate manifest lists every distributed file and an inclusion decision. An exact SHA-256 inventory, outside the candidate tree, binds all files including the manifest itself. A manifest cannot contain its own hash without a defined external construction; its own record explicitly uses the external inventory instead. The private review package contains exact upstream source references and transformations, separately from the intended public files.

Historical case studies are reported from preserved source documents, not recomputed. Reproducing those empirical studies would require lawful versioned inputs, original frozen parameters and methods, complete trial accounting and independent authority. None of that work is authorized or executed by this candidate.

When continuing independently, record input hashes, acquisition/rights status, code identity, environment, complete trial count, costs, decision criteria and artifacts before interpreting results. Hashes establish byte identity, not scientific validity or copyright ownership.
