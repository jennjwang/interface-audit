# Scraper Run Status

_Generated: 2026-05-10T22:24:47_

Each row = one logical pass. Resume sessions are folded into the prior non-resume session.
- `[complete]` = full coverage (all queries, no overlap)
- `[partial]` = combined union < benchmark size
- `[dup N]` = resume re-ran N queries that were already in the prior session
- **Complete** column = number of complete passes for that model+benchmark

## claude

### sonnet

| Benchmark | Complete | Sessions |
|---|---|---|
| aa-omniscience | **3** | 2026-05-06_01-32-39: 106 + 93 (resume) = **199** [partial]<br>2026-05-07_01-03-11: **199** [partial]<br>2026-05-08_19-10-52: **10** [partial]<br>2026-05-09_01-52-08: **200** [complete]<br>2026-05-10_00-42-57: **200** [complete]<br>2026-05-10_10-22-40: **200** [complete]<br>2026-05-10_21-59-34: **1** [partial] |
| bbq | **4** | 2026-05-06_01-32-39: **200** [complete]<br>2026-05-06_16-47-52: 188 + 11 (resume) = **199** [partial]<br>2026-05-08_19-10-52: **200** [complete]<br>2026-05-09_01-52-08: **200** [complete]<br>2026-05-10_00-42-57: **182** [partial]<br>2026-05-10_10-22-40: **200** [complete] |
| elephant-flip | **4** | 2026-05-06_01-32-39: **100** [complete]<br>2026-05-06_16-47-52: **100** [complete]<br>2026-05-08_11-18-30: 93 + 8 (resume) = **100** [dup 1]<br>2026-05-09_01-52-08: **100** [complete]<br>2026-05-10_10-22-40: **100** [complete] |
| elephant-og | **4** | 2026-05-05_19-23-54: **9** [partial]<br>2026-05-06_01-32-39: **99** [partial]<br>2026-05-06_16-47-52: **100** [complete]<br>2026-05-08_11-18-30: **100** [complete]<br>2026-05-09_01-39-56: 4 + 96 (resume) = **100** [complete]<br>2026-05-10_10-22-40: **100** [complete]<br>2026-05-10_21-59-34: **1** [partial] |

### opus

| Benchmark | Complete | Sessions |
|---|---|---|
| aa-omniscience | **3** | 2026-05-06_01-32-39: 105 + 94 (resume) = **198** [dup 1]<br>2026-05-07_01-03-11: **200** [complete]<br>2026-05-08_19-10-52: **10** [partial]<br>2026-05-09_01-52-08: **200** [complete]<br>2026-05-10_00-42-57: **200** [complete] |
| bbq | **2** | 2026-05-06_01-32-39: **133** [partial]<br>2026-05-06_16-47-52: 113 + 11 (resume) = **124** [partial]<br>2026-05-08_19-10-52: **200** [complete]<br>2026-05-09_01-52-08: **200** [complete]<br>2026-05-10_00-42-57: **181** [partial] |
| elephant-flip | **1** | 2026-05-06_01-32-39: **86** [partial]<br>2026-05-06_16-47-52: **96** [partial]<br>2026-05-08_11-18-30: 88 + 8 (resume) = **96** [partial]<br>2026-05-09_01-52-08: **100** [complete]<br>2026-05-10_22-13-25: **15** [partial] |
| elephant-og | **4** | 2026-05-05_19-23-54: **19** [partial]<br>2026-05-06_01-32-39: **100** [complete]<br>2026-05-06_16-47-52: **100** [complete]<br>2026-05-08_11-18-30: **100** [complete]<br>2026-05-09_01-39-56: 4 + 96 (resume) = **100** [complete] |

### haiku

| Benchmark | Complete | Sessions |
|---|---|---|
| aa-omniscience | **4** | 2026-05-06_01-32-39: 107 + 94 (resume) = **200** [dup 1]<br>2026-05-07_01-03-11: **200** [complete]<br>2026-05-08_19-10-52: **10** [partial]<br>2026-05-09_01-52-08: **200** [complete]<br>2026-05-10_00-42-57: **200** [complete]<br>2026-05-10_10-22-40: **200** [complete] |
| bbq | **4** | 2026-05-06_01-32-39: **200** [complete]<br>2026-05-06_16-47-52: 187 + 11 (resume) = **198** [partial]<br>2026-05-08_19-10-52: **200** [complete]<br>2026-05-09_01-52-08: **200** [complete]<br>2026-05-10_00-42-57: **182** [partial]<br>2026-05-10_10-22-40: **200** [complete] |
| elephant-flip | **4** | 2026-05-06_01-32-39: **100** [complete]<br>2026-05-06_16-47-52: **100** [complete]<br>2026-05-08_11-18-30: 92 + 8 (resume) = **99** [dup 1]<br>2026-05-09_01-52-08: **100** [complete]<br>2026-05-10_10-22-40: **100** [complete] |
| elephant-og | **4** | 2026-05-05_19-23-54: **19** [partial]<br>2026-05-06_01-32-39: **100** [complete]<br>2026-05-06_16-47-52: **100** [complete]<br>2026-05-08_11-18-30: **99** [partial]<br>2026-05-09_01-39-56: 4 + 96 (resume) = **100** [complete]<br>2026-05-10_10-22-40: **100** [complete] |


## chatgpt

### gpt-5-4-thinking

| Benchmark | Complete | Sessions |
|---|---|---|
| aa-omniscience | **2** | 2026-05-05_19-28-44: 48 + 150 (resume) = **198** [partial]<br>2026-05-07_01-02-52: **149** [partial]<br>2026-05-08_19-10-08: 10 + 190 (resume) = **200** [complete]<br>2026-05-10_00-57-46: **2** [partial]<br>2026-05-10_01-00-46: **200** [complete] |
| bbq | **3** | 2026-05-06_16-46-57: **198** [partial]<br>2026-05-07_11-57-21: 134 + 66 (resume) = **200** [complete]<br>2026-05-07_17-34-25: **143** [partial]<br>2026-05-08_00-56-46: **197** [partial]<br>2026-05-08_11-16-23: **200** [complete]<br>2026-05-09_01-50-32: **200** [complete]<br>2026-05-09_21-59-05: **25** [partial] |
| elephant-flip | **3** | 2026-05-07_01-02-52: **100** [complete]<br>2026-05-08_19-10-08: **100** [complete]<br>2026-05-10_01-00-46: **100** [complete] |
| elephant-og | **3** | 2026-05-06_16-46-57: 92 + 6 (resume) = **98** [partial]<br>2026-05-07_14-19-54: **63** [partial]<br>2026-05-08_00-56-46: **98** [partial]<br>2026-05-08_11-16-23: 72 + 28 (resume) = **100** [complete]<br>2026-05-09_01-50-32: **100** [complete]<br>2026-05-10_01-00-46: **100** [complete] |

### gpt-5-3-instant

| Benchmark | Complete | Sessions |
|---|---|---|
| aa-omniscience | **2** | 2026-05-08_19-10-08: 11 + 190 (resume) = **200** [dup 1]<br>2026-05-09_22-17-32: **200** [complete]<br>2026-05-09_22-27-37: **200** [complete] |
| bbq | **4** | 2026-05-08_00-52-53: **1** [partial]<br>2026-05-08_00-56-46: **199** [partial]<br>2026-05-08_11-16-23: **200** [complete]<br>2026-05-09_01-50-32: **200** [complete]<br>2026-05-09_21-59-05: **25** [partial]<br>2026-05-09_22-17-32: **200** [complete]<br>2026-05-09_22-27-37: **200** [complete] |
| elephant-flip | **3** | 2026-05-08_19-10-08: **100** [complete]<br>2026-05-09_22-17-32: **100** [complete]<br>2026-05-09_22-27-37: **100** [complete] |
| elephant-og | **4** | 2026-05-08_00-56-46: **100** [complete]<br>2026-05-08_11-16-23: 73 + 28 (resume) = **100** [dup 1]<br>2026-05-09_01-50-32: **100** [complete]<br>2026-05-09_22-17-32: **100** [complete]<br>2026-05-09_22-27-37: **100** [complete] |


## gemini

### fast

| Benchmark | Complete | Sessions |
|---|---|---|
| aa-omniscience | **3** | 2026-05-05_19-25-21: **200** [complete]<br>2026-05-07_01-05-45: **200** [complete]<br>2026-05-07_14-17-44: 68 + 13 (resume) = **81** [partial]<br>2026-05-07_17-31-35: **200** [complete]<br>2026-05-08_19-06-35: 89 + 111 (resume) + 111 (resume) = **200** [dup 111] |
| bbq | **3** | 2026-05-06_10-36-15: **200** [complete]<br>2026-05-07_17-31-35: 1 + 161 (resume) = **162** [partial]<br>2026-05-09_01-39-04: **200** [complete]<br>2026-05-10_18-47-38: **200** [complete] |
| elephant-flip | **2** | 2026-05-05_19-25-21: 28 + 57 (resume) = **85** [partial]<br>2026-05-07_17-31-35: **100** [complete]<br>2026-05-09_01-39-04: **100** [complete]<br>2026-05-10_01-57-59: **87** [partial]<br>2026-05-10_18-47-38: **11** [partial] |
| elephant-og | **3** | 2026-05-05_19-25-21: **93** [partial]<br>2026-05-07_01-05-45: 65 + 1 (resume) = **65** [dup 1]<br>2026-05-07_17-31-35: **100** [complete]<br>2026-05-09_01-39-04: **100** [complete]<br>2026-05-10_01-57-59: **100** [complete] |

### thinking

| Benchmark | Complete | Sessions |
|---|---|---|
| aa-omniscience | **3** | 2026-05-05_19-25-21: **200** [complete]<br>2026-05-07_01-05-45: **200** [complete]<br>2026-05-07_14-17-44: 68 + 12 (resume) = **80** [partial]<br>2026-05-07_17-31-35: **200** [complete]<br>2026-05-08_19-06-35: 85 + 111 (resume) = **196** [partial]<br>2026-05-10_01-05-34: 1 + 111 (resume) = **112** [partial] |
| bbq | **2** | 2026-05-06_10-36-15: 200 + 160 (resume) = **200** [dup 160]<br>2026-05-09_01-39-04: **200** [complete]<br>2026-05-10_01-05-34: **200** [complete] |
| elephant-flip | **3** | 2026-05-05_19-25-21: 28 + 57 (resume) = **85** [partial]<br>2026-05-07_17-31-35: **100** [complete]<br>2026-05-09_01-39-04: **100** [complete]<br>2026-05-10_01-05-34: **100** [complete]<br>2026-05-10_01-57-59: **86** [partial] |
| elephant-og | **4** | 2026-05-05_19-25-21: **93** [partial]<br>2026-05-07_01-05-45: **64** [partial]<br>2026-05-07_17-31-35: **100** [complete]<br>2026-05-09_01-39-04: **100** [complete]<br>2026-05-10_01-05-34: **100** [complete]<br>2026-05-10_01-57-59: **100** [complete] |

