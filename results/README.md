# Recorded results

`summary.json` contains the numerical conclusions from the executed research run.
`figure_data.json` contains exact measured plotting inputs, including standard errors over seed-wise fixed-family averages.
`environment_all.json`, `independent_theory_summary.json`, and `test_report.json` record execution provenance and checks.

The complete downloadable research bundle accompanying the paper additionally contains all raw CSV/JSON files and a checksummed `raw_records.tar.xz` archive, including seed-zero model weights and discrepancy tables. The raw archive is not stored in this GitHub tree. From a checkout, `make experiments` regenerates those records; `make paper` builds directly from the committed tables and measured figure inputs without retraining.

Theoretical validation is exhaustive only in the explicitly reported four-component finite class. Forty successful confidence-wrapper runs do not by themselves establish statistical validity. All mechanistic completeness statements remain conditional on the hypothesis class and intervention policy.
