# What-If Engine · Limitations

Honest, sharp, unshied. The exact list a hostile reviewer would
compile — better to write it here than let them find it.

## Data

- **IMD sentinel confusion is real.** Rain files use `-999.0`, temp
  files use `99.9`. The engine masks both at the L0 reader boundary
  (`whatif/drivers/_common.py::mask_sentinels`). Any downstream code
  that reads raw grids and skips the mask will silently produce
  garbage.
- **INSAT record spans only ~2 years** in the processed cube (2023 to
  2025). Any fused-encoder contribution is fundamentally small. Fix
  requires longer INSAT archives — data problem, not code.
- **ERA5 not ingested.** Reference ET0 is Hargreaves-Samani, not
  Penman-Monteith. FAO-56 §3 accepts this fallback but PM would be
  more accurate over the humid southwestern coast. Fix: `et0_penman.py`
  is stubbed and awaits an ERA5 ingest.
- **APY district yields have known reporting lags** (~2 years for
  many states) and inconsistent series across states before 2010.
  The validation module (`whatif/sectors/validation_apy.py`) refuses
  to fabricate a validation number when the parquet isn't on disk.
- **NEX-GDDP-CMIP6 is remote.** Without `NEX_GDDP_ROOT` set or
  `fsspec + s3fs` installed, Long-Term panels surface an
  "illustrative until data on disk" banner and refuse to fabricate
  the exact return-period or ToE numbers.

## Model

- **ConvLSTM at 0.25° cannot resolve orographic gradients in the
  Western Ghats.** Sub-grid variability shows up as bias in the ET0
  and dry-spell indices there. Fix requires higher-resolution
  downscaling (a next-release item).
- **MC-Dropout is a computationally cheap approximation** to a full
  deep ensemble. It captures parametric uncertainty at O(K forward
  passes) but under-represents structural uncertainty. The ensemble
  driver stamps `source="mausamsetu_ensemble"` — never claims full
  ensemble diversity.
- **Bias correction against IMD may propagate IMD's own gauge-density
  biases.** Regions with sparse gauges (Arunachal, Northeast) inherit
  the interpolation smoothing.

## Economics

- **MSP is a floor, not a market price.** Payoff numbers computed at
  MSP are the price a farmer is guaranteed, not what they receive on
  a mandi day. The optional Agmarknet lever (Part 7 stub) widens the
  band; it is off by default and its raising `NotImplementedError`
  makes silent fallback impossible.
- **Cost-of-cultivation figures are national averages** with regional
  variation not modelled. A per-state cost table would tighten the
  numbers; not shipped in this release.
- **Discount rate is a policy choice, not a physical constant.** The
  8 % central rate follows NITI Aayog; the reported 7–12 % band exists
  because reasonable people disagree.

## Analog method

- **Van den Dool 1994 governs.** Perfect analogs are rare in a 73-year
  record. Rule 7 mandates honest reporting — quality tiers are shown
  as `strong / fair / poor` and many regions × windows will have no
  `strong` match. Scenario 3 in the demo set is a shipped
  no-strong-analog example.
- **Mahalanobis distance handles feature correlation but not
  non-Gaussianity.** Features are standardised on TRAIN_YEARS, so the
  correlation structure is representative — but tail behaviour of the
  feature vector isn't Gaussian and the χ² tier cutoffs are a
  heuristic anchor, not a strict test.

## Long Term

- **Multi-model spread widens with horizon. This is physics, not a
  bug.** The Hawkins-Sutton panel makes this explicit by decomposing
  scenario × model × internal fractions.
- **SSP pathways are policy scenarios, not predictions.** The Long
  Term tab's permanent banner enforces this framing; the copy lint
  refuses `predict / forecast / will / is going to`.
- **QDM assumes stationarity of the bias-correction transfer
  function.** If the observed baseline period (1971-2000) does not
  span the range of physical regimes seen in the future, QDM will
  extrapolate — a known limitation of all delta-family downscaling.

## UI

- **Streamlit reruns the whole page on interaction.** The
  `state.cache_key()` design keeps engine calls from re-running when
  UI-only preferences change; complex compositions with many widget
  interactions can still lag on the first-render pass. The `slow`
  pytest marker covers the AppTest smoke.
- **Landing card on `Home.py` is deferred.** The app's entry point in
  this branch is `app_v2.py` (with monkey-patched pages); no
  `Home.py` file exists, so Part 8 STEP 9 is a documented deferral,
  not a bug.

## Coverage

- **Sectors not yet in the release:**
  - Energy (peak-load × Tmax anomaly; coal displacement from PV +
    storage) — Part 8 stretch / next release. Demo scenario 08 ships
    the YAML with `status: deferred`.
  - Water — reservoir routing; irrigation demand. Part 8 stretch.
  - Health — WBGT + labour productivity, heat-stroke mortality
    valuation. Part 7 spec listed this; deferred.
  - Disaster — cyclone tracks + damage functions; only the
    return-period shift diagnostic (Part 7 STEP 5a) is currently
    plumbed. Demo scenario 06 exercises the plumbing.

## Fundamentally unresolvable limits

- **The framework's confidence in Long Term is inter-model agreement
  and emergence significance, not skill against out-of-sample
  climate.** No amount of engineering fixes this; it is what "future
  scenario" means. Rule 10.
- **The what-if framework cannot enumerate all levers.** It answers
  well-posed decision questions and refuses to be a general-purpose
  chatbot.
