# Iridium Dataset Description

## Dataset overview

- **Data directory:** `/Users/sanyaarora/Desktop/Surrey/Iridium_Satellite/data/raw`
- **Total number of messages:** 50,000
- **Number of unique satellites:** 121
- **Number of metadata columns:** 23

## Dataset columns

| Column | Number of segments | Shape per segment | Data type | Total size (MB) |
|---|---:|---|---|---:|
| `bytes` | 5 | [10000, 111] | uint8 | 5.3 |
| `center_frequency` | 5 | [10000] | int64 | 0.4 |
| `confidence` | 5 | [10000] | int64 | 0.4 |
| `direction` | 5 | [10000] | int64 | 0.4 |
| `level` | 5 | [10000] | float64 | 0.4 |
| `magnitude` | 5 | [10000] | float64 | 0.4 |
| `msg` | 5 | [10000] | <U562 | 107.0 |
| `msg_id` | 5 | [10000] | int64 | 0.4 |
| `msg_type` | 5 | [10000] | <U2 | 0.4 |
| `n_symbols` | 5 | [10000] | int64 | 0.4 |
| `noise` | 5 | [10000] | float64 | 0.4 |
| `ra_alt` | 5 | [10000] | float64 | 0.4 |
| `ra_cell` | 5 | [10000] | int64 | 0.4 |
| `ra_lat` | 5 | [10000] | float64 | 0.4 |
| `ra_lon` | 5 | [10000] | float64 | 0.4 |
| `ra_sat` | 5 | [10000] | int64 | 0.4 |
| `run_id` | 5 | [10000] | int64 | 0.4 |
| `sample_count` | 5 | [10000] | int64 | 0.4 |
| `sample_rate` | 5 | [10000] | int64 | 0.4 |
| `samples` | 5 | [10000, 11000, 2] | float32 | 4196.2 |
| `timestamp` | 5 | [10000] | int64 | 0.4 |
| `timestamp_global` | 5 | [10000] | int64 | 0.4 |
| `uw_start` | 5 | [10000] | float64 | 0.4 |


## Top 10 most-sampled satellites

| Rank | Satellite ID | Message count |
|---:|---:|---:|
| 1 | 92 | 1,282 |
| 2 | 85 | 1,277 |
| 3 | 87 | 1,232 |
| 4 | 51 | 1,188 |
| 5 | 109 | 1,184 |
| 6 | 93 | 1,171 |
| 7 | 24 | 1,159 |
| 8 | 111 | 1,157 |
| 9 | 38 | 1,149 |
| 10 | 28 | 1,131 |


## Satellite distribution

- **Mean messages per satellite:** 413
- **Maximum messages for one satellite:** 1,282
- **Minimum messages for one satellite:** 1

## One IQ sample

A single IQ sample represents one Iridium Ring Alert message header.

Each sample has shape
`[11000, 2]`
and dtype `float32`.

The shape `[11000, 2]` means that each message contains:

- 11,000 time-domain sample points
- two values per point: the in-phase component `I` and quadrature component `Q`

## Dataset size

- **Number of messages:** 50,000
- **Number of unique satellites:** 121
- **Total extracted disk size:** approximately
  4316 MB
