# P1-M5 scoped P2 inventory supplement (metadata-only)

The immutable M5 source-revision registration is the sole inventory source.
The general closing package in OpenHyra commit `0bcfbbb` first enriches the
validated 20-day receipt with a pinned dataset tree inventory. Its 2,303
complete three-table dates span June 2017 through November 2026. A complete
stock-presence audit over every date would require up to 20,727 separate HEAD
and exact-range requests, so stock membership is audited only for the 147
market dates from 2017-06-01 through 2017-12-31, satisfying the 2017 second-half
scope. The remaining 2,156 dates are recorded as uninspected and unknown.

With Python 3.12, committed closure code `p1-m5-closing-code-20261126`, its
matching Stochopia checkout, and the original authorized Hugging Face tools on
`PYTHONPATH`, execute (only after the 90-slot seal and validated acquisition
receipt exist):

```sh
python -m experiments.p1_m5_closing_data.scoped_footer_audit \
  --artifact "$M5_ARTIFACT_ROOT" \
  --progress "$PRIVATE_ROOT/footer-progress-2017.json" \
  --through 20171231 --workers 6
```

The progress file is a private resumable record of one classification per date
and table; no data rows, signed URLs, or credentials are logged. The script uses
committed `file_presence` for retrieval rather than an alternate method. Each
file request obtains metadata length, fetches the exact last eight bytes and
the exact Parquet metadata footer in HTTP 206 range requests, rejects any full
file/non-exact response, parses the footer only, and reads only `wind_code`
row-group min/max/count metadata. Given fixed-length Wind identifiers, endpoint
matches certify presence; interior bounds remain unresolved. Every stock-day
needs confirmed presence in all three tables; any proven absence excludes that
date. Metadata errors are recorded as unknown. No market data page was requested.

After all 441 table classifications, the script validates the original receipt
again, checks that all its underlying data fields and remote tree inventory
remain byte-semantically unchanged, and atomically updates `remote_availability`
under the acquisition lock. New `footer_audit.by_stock` entries show exact or
bounded stock-specific counts and dates for the June–December 2017 scope.
Top-level `by_stock` combines those results with explicitly unknown remaining
market dates, rather than misrepresenting market-wide coverage as stock coverage.

## Actual observations

- All 147 date bundles and 441 table metadata requests successfully classified;
  zero transport/metadata failures.
- `000001.SZ`: exact 146 complete stock-day bundles; 2017-12-01 absent in all
  three table metadata panels.
- `300001.SZ`: 137 confirmed full bundles; 147 possible; 10 unresolved dates
  because the security code lies inside the row-group bounds, not at an endpoint.
- Both symbols have confirmed coverage in every month June–December 2017.
- Beyond 2017: all 2,156 dates uninspected. Full dataset bounds are 146–2302
  and 137–2303 confirmed/possible complete stock-day bundles respectively.

These are file-membership results, not verification of a date's valid bilateral
quoting anchors, suspension status, or precise microstructure content.
