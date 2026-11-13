# CarRacing-v3: before vs. after training

10 episodes per policy on the same tracks (seeds 10042-10051), scored with the unmodified environment reward. "After" is `models/ppo_carracing/final_model.zip`, acting deterministically.

## Score

| Metric | Before (random) | After (trained PPO) | Change |
|---|---|---|---|
| Mean return | -43.0 | 917.0 | +960.0 |
| Std of return | 11.6 | 9.6 | -2.0 |
| Median return | -38.6 | 918.7 | +957.3 |
| Min return | -66.6 | 894.6 | +961.2 |
| Max return | -29.3 | 928.5 | +957.8 |
| Episodes scoring 900+ | 0 / 10 | 9 / 10 | +9 |

## Driving

| Metric | Before (random) | After (trained PPO) | Change |
|---|---|---|---|
| Laps completed | 0 / 10 | 9 / 10 | +9 |
| Drove off the map | 0 / 10 | 0 / 10 | +0 |
| Mean track completion (%) | 5.9 | 100.0 | +94.0 |
| Lowest track completion (%) | 3.5 | 99.7 | +96.2 |
| Mean lap time (s) | n/a | 14.6 | n/a |
| Fastest lap (s) | n/a | 12.6 | n/a |
| Mean episode length (agent steps) | 475.0 | 376.4 | -98.6 |
| Mean episode length (s) | 19.0 | 15.1 | -3.9 |
| Mean speed (simulator units) | 2.8 | 63.5 | +60.7 |
| Top speed (simulator units) | 16.3 | 91.1 | +74.8 |
| Time with all wheels off the road (%) | 5.3 | 0.0 | -5.3 |

## Action usage (% of steps)

| Action | Before (random) | After (trained PPO) |
|---|---|---|
| nothing | 20.5 | 5.7 |
| steer right | 19.5 | 5.3 |
| steer left | 20.7 | 13.4 |
| gas | 19.6 | 74.9 |
| brake | 19.7 | 0.7 |

## Per track

| Seed | Track tiles | Before return | After return | Before completion (%) | After completion (%) | After result | After time (s) |
|---|---|---|---|---|---|---|---|
| 10042 | 317 | -66.6 | 908.2 | 3.5 | 100.0 | lap completed | 17.1 |
| 10043 | 249 | -34.8 | 924.1 | 6.8 | 100.0 | lap completed | 13.6 |
| 10044 | 287 | -42.7 | 916.7 | 5.9 | 100.0 | lap completed | 15.3 |
| 10045 | 257 | -40.5 | 923.6 | 6.2 | 100.0 | lap completed | 13.7 |
| 10046 | 273 | -36.4 | 918.5 | 6.6 | 100.0 | lap completed | 14.8 |
| 10047 | 301 | -61.8 | 911.6 | 4.0 | 100.0 | lap completed | 16.4 |
| 10048 | 288 | -46.4 | 894.6 | 5.6 | 99.7 | ran out of time | 19.0 |
| 10049 | 249 | -34.8 | 925.8 | 6.8 | 100.0 | lap completed | 13.2 |
| 10050 | 274 | -29.3 | 919.0 | 7.3 | 100.0 | lap completed | 14.8 |
| 10051 | 240 | -36.7 | 928.5 | 6.7 | 100.0 | lap completed | 12.6 |

## How to read this

- **Return** is the CarRacing score: +1000/N for each of the track's N tiles reached, -0.1 per simulator frame, -100 for driving off the map.
- **Episodes scoring 900+**: 900 is the average score at which CarRacing counts as solved.
- **Lap completed** means the car got back to the start having reached more than 95% of the tiles, which ends the episode early. A random policy never gets that far, so its episodes run to the time limit.
- **Track completion** is tiles reached / tiles on the track.
- **Times** count from the moment the agent takes control, after the 1-second opening zoom. One agent step is 0.04 s.
- **Speed** is in the simulator's own units, the quantity CarRacing draws as its speed bar.
- **All wheels off the road** is the share of steps on which no wheel touched a road tile.
