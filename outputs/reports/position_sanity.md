# Position field sanity check

## Why this check was performed

48.6% of decoded positions place the satellite below the receiver's
horizon, which cannot occur for a message that was received. Decode
confidence does not explain the discrepancy (Mann-Whitney p = 0.17) and the
values are structured rather than random, so corruption is not a sufficient
explanation.

The remaining possibility is that `ra_lat`/`ra_lon`/`ra_alt` do not
describe the satellite named by `ra_sat`. If that were so, the class labels
would not identify the transmitter and the classification task would be
mislabelled.

## Method

A satellite in low Earth orbit moves at approximately
7.5 km/s, so between consecutive messages separated by dt
seconds its sub-point can travel at most about 7.5 * dt km.
Implied speeds were computed for consecutive messages from the same
satellite within a single pass (gaps under 600 s).

## Results

| Satellite | Messages | Pairs | Median implied speed (km/s) | Within 1.5x orbital velocity |
|----------:|---------:|------:|----------------------------:|-----------------------------:|
| 92 | 1,282 | 1,272 | 348.40 | 13.8% |
| 85 | 1,277 | 1,264 | 332.75 | 15.3% |
| 87 | 1,232 | 1,220 | 333.96 | 14.9% |
| 51 | 1,188 | 1,178 | 294.21 | 15.4% |
| 109 | 1,184 | 1,176 | 329.15 | 13.9% |

Overall median implied speed: **331.97 km/s** against an orbital
velocity of 7.5 km/s. 14.7% of consecutive pairs
fall within 1.5 times orbital velocity.

## Conclusion

Implied speeds are largely inconsistent with orbital motion. The position field does not describe the trajectory of the satellite named by ra_sat, and the meaning of both fields requires re-examination.
