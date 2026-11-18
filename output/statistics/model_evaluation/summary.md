# Automatic model evaluation

## Test set: chat (25 conversations)

| Metric | real replies | base | full chat dump | masked chat dump q k | masked chat dump q k v o | synthetic chat | Best is |
|---|---:|---:|---:|---:|---:|---:|---|
| Test loss | – | 4.105 (3.690 – 4.501) | 3.320 (2.957 – 3.636) | 3.371 (3.018 – 3.690) | **3.277 (2.954 – 3.563)** | 3.314 (2.948 – 3.636) | lower |
| Perplexity | – | 55.8 | 34.4 | 36.2 | **33.7** | 33.7 | lower |
| Next-token accuracy % | – | 34.5 (28.2 – 41.4) | 37.2 (31.0 – 43.9) | 37.8 (31.8 – 44.4) | 38.8 (33.2 – 45.4) | **39.2 (33.3 – 45.6)** | higher |
| chrF | – | **13.2 (10.5 – 15.7)** | 9.7 (8.0 – 11.3) | 10.2 (8.4 – 11.8) | 9.0 (7.5 – 10.5) | 11.7 (9.5 – 13.6) | higher |
| ROUGE-L | – | 4.4 (3.0 – 5.8) | 5.0 (3.1 – 7.2) | 4.2 (2.5 – 5.8) | 4.3 (2.5 – 6.2) | **5.9 (4.0 – 7.8)** | higher |
| Median words per reply | 10 | 64 | 11 | **10** | 9 | 11 | closest to real |
| Lines per reply | 1.7 | 7.0 | 2.5 | 2.0 | **1.6** | 1.8 | closest to real |
| Lines starting lowercase % | 32 | 19 | 61 | 52 | **29** | 69 | closest to real |
| Lines ending with a period % | 12 | 19 | **10** | 28 | 6 | 0 | closest to real |
| Emotes per 100 words | 2.4 | 7.1 | 0.7 | 0.9 | 1.2 | **1.8** | closest to real |
| Replies with markup (* # `) % | 4 | 37 | **0** | 0 | 0 | 0 | closest to real |
| Word distribution distance (JSD) | – | 0.57 | **0.54** | 0.54 | 0.54 | 0.56 | lower |
| Length distribution distance (KS) | – | 0.80 | **0.15** | 0.19 | 0.15 | 0.21 | lower |
| Replies cut at token limit % | – | 48 (37 – 59) | 4 (0 – 8) | 3 (0 – 7) | **0 (0 – 0)** | 1 (0 – 4) | lower |
| Repeated 3-grams in a reply % | 0.0 | 4.5 | 0.9 | 2.3 | **0.7** | 1.0 | closest to real |
| Echo of the user's message (chrF) | 12.0 | 20.3 (17.9 – 22.9) | 13.8 (11.8 – 15.9) | 15.0 (12.4 – 17.8) | **12.9 (10.4 – 15.9)** | 21.6 (15.6 – 29.6) | closest to real |
| Lines copied from training % | – | – | **0.0** | 0.0 | 0.0 | 0.9 | lower |

Loss difference to the base model (paired, per conversation; negative = better):

- full chat dump: -0.786 (-1.068 – -0.545), better on 100% of conversations, significant at 95%
- masked chat dump q k: -0.734 (-0.997 – -0.501), better on 100% of conversations, significant at 95%
- masked chat dump q k v o: -0.829 (-1.135 – -0.566), better on 100% of conversations, significant at 95%
- synthetic chat: -0.791 (-1.046 – -0.573), better on 100% of conversations, significant at 95%

## Test set: synthetic (25 conversations)

| Metric | real replies | base | full chat dump | masked chat dump q k | masked chat dump q k v o | synthetic chat | Best is |
|---|---:|---:|---:|---:|---:|---:|---|
| Test loss | – | 5.290 (4.801 – 5.805) | 3.545 (3.206 – 3.909) | 3.791 (3.466 – 4.150) | 3.545 (3.227 – 3.903) | **3.357 (2.987 – 3.733)** | lower |
| Perplexity | – | 111.8 | 37.5 | 45.5 | 37.0 | **32.5** | lower |
| Next-token accuracy % | – | 27.5 (22.9 – 32.0) | 33.2 (27.5 – 38.8) | 29.4 (24.4 – 34.6) | 34.3 (28.4 – 40.0) | **34.7 (28.5 – 41.1)** | higher |
| chrF | – | **14.7 (11.2 – 18.0)** | 13.9 (10.7 – 17.0) | 13.5 (10.5 – 16.4) | 11.3 (9.1 – 13.3) | 13.2 (10.6 – 15.5) | higher |
| ROUGE-L | – | 5.6 (4.1 – 7.2) | **9.1 (6.1 – 13.0)** | 7.0 (4.9 – 9.3) | 5.2 (3.7 – 6.8) | 6.4 (4.6 – 8.4) | higher |
| Median words per reply | 14 | 80 | **16** | 28 | 17 | 16 | closest to real |
| Lines per reply | 2.1 | 8.0 | 3.5 | 3.5 | 2.7 | **2.3** | closest to real |
| Lines starting lowercase % | 18 | 0 | 49 | 2 | 2 | **23** | closest to real |
| Lines ending with a period % | 0 | 18 | 23 | 62 | 54 | **9** | closest to real |
| Emotes per 100 words | 3.0 | 6.4 | 0.2 | **0.4** | 0.2 | 0.1 | closest to real |
| Replies with markup (* # `) % | 0 | 67 | 1 | 16 | **0** | 0 | closest to real |
| Word distribution distance (JSD) | – | 0.58 | 0.53 | 0.58 | 0.56 | **0.52** | lower |
| Length distribution distance (KS) | – | 0.91 | **0.15** | 0.33 | 0.16 | 0.17 | lower |
| Replies cut at token limit % | – | 56 (43 – 69) | 7 (1 – 12) | 24 (12 – 39) | 8 (1 – 16) | **0 (0 – 0)** | lower |
| Repeated 3-grams in a reply % | 0.0 | **1.0** | 4.3 | 4.8 | 4.4 | 3.0 | closest to real |
| Echo of the user's message (chrF) | 18.9 | 23.3 (20.3 – 26.5) | 19.5 (16.4 – 22.9) | 24.0 (19.1 – 30.2) | **18.6 (15.8 – 22.1)** | 35.8 (28.8 – 43.6) | closest to real |
| Lines copied from training % | – | – | **0.0** | 0.0 | 0.0 | 0.0 | lower |

Loss difference to the base model (paired, per conversation; negative = better):

- full chat dump: -1.745 (-2.138 – -1.369), better on 100% of conversations, significant at 95%
- masked chat dump q k: -1.499 (-1.875 – -1.140), better on 100% of conversations, significant at 95%
- masked chat dump q k v o: -1.745 (-2.150 – -1.360), better on 100% of conversations, significant at 95%
- synthetic chat: -1.933 (-2.347 – -1.540), better on 100% of conversations, significant at 95%

## Test-set leakage

Share of test replies that occur verbatim in the model's own training data. "Trained on" = inside text the loss was computed on; "in training text" also counts prompts.

| Model | Test set | Trained on % | In training text % |
|---|---|---:|---:|
| full chat dump | chat | 8 | 8 |
| full chat dump | synthetic | 72 | 72 |
| masked chat dump q k | chat | 8 | 8 |
| masked chat dump q k | synthetic | 60 | 72 |
| masked chat dump q k v o | chat | 8 | 8 |
| masked chat dump q k v o | synthetic | 60 | 72 |
| synthetic chat | chat | 88 | 88 |
| synthetic chat | synthetic | 16 | 16 |

Intervals are 95% percentile bootstrap intervals over test conversations (5000 resamples). Generation metrics average 3 sampled replies per prompt.
