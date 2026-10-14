# Security notes

MRI is a public, read-only demo. It serves one dataset that the app generates for
itself, has no accounts, no login, no user-supplied files and no write path to any
store a visitor can reach. The notes below record how the OWASP Top 10 (2021) was
applied to that design, and where a control is **not applicable** rather than
implemented.

## Reporting

Open an issue, or email the address on the repository owner's GitHub profile. No
bounty is offered. Since the app holds no real data, please do not run automated
scanners against the hosted demo.

## Injection (A03)

This is the control the project cares most about, because the pipeline is
SQL-driven end to end.

- **Every write is parameterised.** `mri.governance.ops` inserts into
  `pipeline_runs`, `model_registry`, `governance_artifacts` and `drift_reports`
  using `?` placeholders. No value is ever formatted into a statement.
- **Every read is a module-level literal.** `mri.features.engineering`,
  `mri.detection.graph` and `mri.detection.temporal` pass constant `SELECT`
  strings. `mri.app.data.REGISTRY_QUERIES` holds the three registry statements as
  literals, so no table name is interpolated either.
- **A guard enforces that.** `_check_query()` in the three pipeline modules
  rejects anything that is not a single `SELECT`, so a future caller cannot
  smuggle a second statement through the helper.
- **Visitor input never reaches SQL.** The only inputs in the UI are the band
  pills, the entity-type control, a slider and an ID search box. They filter an
  in-memory pandas DataFrame. The search uses `str.contains(..., regex=False)`,
  so a crafted string cannot become a pattern or cause catastrophic backtracking.
- **No shell.** The one subprocess call (`mri.pipeline` on first launch) uses a
  fixed `argv` list with `shell=False` and no interpolated input.
- **No HTML injection.** Views render through `st.html`. Every interpolated value
  passes through `theme.esc()` (`html.escape(..., quote=True)`) first.
- **No `eval`, `exec`, or dynamic import** anywhere in `src/`.

## A01 Broken access control

No roles and no per-user data: every visitor sees the same generated dataset. The
app reads from fixed paths in `mri.config`; no path is built from input, so there
is no traversal surface. The one download (`case_<entity>.json`) is assembled in
memory and its filename is passed through a `[^A-Za-z0-9_-]` filter.

## A02 Cryptographic failures

No secrets, credentials, tokens or personal data exist in the repository or the
running app, so there is nothing to encrypt at rest. Streamlit Community Cloud
terminates TLS for the hosted demo. SHA-256 is used for artefact integrity, not
for secrecy.

## A04 Insecure design

The threat model is deliberately small: public data in, nothing written back out.
The risk that remains is **misinterpretation**, not compromise — someone reading
perfect model metrics as evidence of real detection. The About and Model lab pages
state that the data is synthetic, that labels came from the V1 rules, that the
anomaly signal is counted twice, and that the fusion weights were set by hand.

## A05 Security misconfiguration

`.streamlit/config.toml` sets `showErrorDetails = "none"`, so a visitor never sees
a traceback or a server path; `enableXsrfProtection = true`; `maxUploadSize = 1`,
since the app accepts no uploads; and `gatherUsageStats = false`. Generated
artefacts (`data/marketplace.db`, `artifacts/`, `models/`, `outputs/`) are
gitignored, so nothing from a local run is published by accident.

## A06 Vulnerable and outdated components

Dependencies are declared in `requirements.txt` and `pyproject.toml` with floors.
CI runs `pip-audit` on every push and pull request, alongside `bandit -r src -ll`.
Dependencies are limited to the scientific stack plus Streamlit and Plotly; no
package is vendored or installed from a URL.

## A07 Identification and authentication failures

Not applicable: the app has no authentication, no sessions that carry identity and
no password, token or recovery flow.

## A08 Software and data integrity failures

Every model file is registered with its size and SHA-256 hash, so a changed
artefact is detectable on the Governance page. `joblib.load` deserialises pickles
and is therefore only ever pointed at `models/`, which this app's own pipeline
wrote in the same container — never at an uploaded or downloaded file. CDN scripts
are not used; the only external resource is a Google Fonts stylesheet.

## A09 Logging and monitoring failures

`mri.governance.ops` records every stage start and finish in `pipeline_runs`, and
writes a health summary and a drift report each run. Both are surfaced on the
Governance page. Logs carry no personal data. Note the honest limitation already
shown in the UI: the drift baseline is written during the same run, so drift reads
0% until a later run compares against it.

## A10 Server-side request forgery

Not applicable: the application makes no outbound HTTP requests. Nothing in the UI
accepts or follows a URL.

## Known limitations

- Security headers (CSP, HSTS, `X-Frame-Options`) are set by the Streamlit
  Community Cloud edge, not by this application; self-hosting behind a reverse
  proxy would be the place to add them.
- There is no rate limiting. The hosted demo relies on the platform for that.
- The app is single-tenant by design. Adding real marketplace data would require
  authentication, per-tenant isolation and an audit trail that this demo does not
  have.
