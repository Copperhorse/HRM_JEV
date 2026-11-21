# Open-Jev `release-v2-redistributable` vs the 231 public JevBench items

VERIFIED by running the loaders and the HRM-Text tokenizer on the pinned data (2026-11-21); prompts rendered with our PrefixLM template.

| | Open-Jev train / test / OOD | JevBench public (231) |
|---|---|---|
| state form | **JSON/structured for 95% of train** (only `customer-control-v1`, 4,206 of 79,116 rows, is free text); sources are games/geometry/workflow generators | **free text for 85%** (policies, messages, documents); hand-authored |
| kinds | train: noul 63%, choice 23%, score 14% (4k sample) | easy: choice 36 + noul 12; standard: choice 36, noul 24, score 12; hard: choice 67, noul 38, score 6 |
| options | 2 (64%), 3, 4, 5, 7, 9, 16 | 2-6 |
| prompt tokens (median / p90 / p99 / max rows >1000) | train 329 / 636 / 738 / **0%**; test 425 / 654 / 765; OOD 434 / 665 / 786 | easy 111 / 131 / 137; standard 107 / 137 / 150; **hard 656 / 2,994 / 4,014, 40% > 1000 tokens, 10% > 3000** |
| noul wording | question + "Is the answer to this question yes?" | same, plus "Yes means: / No means:" lines (0 of 48,907 train noul rows have them) |
| labels | generator rules / oracles / heuristics (`target_basis`) | authored with rationale; hard tier built adversarially (surface answer != expected in 106 of 106 items that have one) |

Source-name overlap with JevBench is nil: JevBench families (policy, intent, extraction, long_policy, multi_hop, temporal_numeric, trap, ...) have no counterpart among the 12 Open-Jev sources.

JevBench accuracy of `seed1_long` by prompt length (tokens): easy [0,200) 48/48; standard [0,200) 53/72; hard [200,500) 24/41, [500,1000) 6/24, [1000,2000) 0/8, [2000,5000) 15/37. Not monotone in length (the long hard items are partly the easier kinds), so length alone is not shown to be the cause. UNVERIFIED hypotheses: the gap comes from form (structured-state rules vs. applying prose policy) and from length/format never seen in training.

Implications:
- Open-Jev test/OOD and JevBench are different distributions; Open-Jev OOD is out-of-distribution by *source*, but still the same structured-state generators and the same length range as train.
- Do NOT train on the 231 public items: they are the evaluation set (JevBench requires disclosing training/selection on them; the 1,200 sealed items would be scored anyway) and 231 rows would not move a 79k-row run.
- Any extra data mixed in changes the recipe (compare on Open-Jev validation/OOD and on a held-out set of the new data, not on JevBench). Long rows are expensive here: ~8 module passes per token, ~740 tok/s on 2×T4, so a 1,300-token row costs roughly 2 s.

## seed1_long per source on Open-Jev (hard acc / NLL at T = 1.0823), VERIFIED from `runs/seed1_long/preds/best`
- Free-text source `customer-control-v1`: test 1.000 / 0.043, OOD 0.982 / 0.111. When prose rows come with the same rule family as in training, the model handles them; the JevBench gap is about *different* tasks (applying authored policy text, long documents), not prose per se.
- **44% of OOD rows (6,912 of 15,701) are `painting-geometry-v1`: OOD hard acc 0.817, NLL 1.77.** That one source drives the OOD NLL/ECE problem; other OOD sources are 0.91-1.00 except `reasoning-control-v1` (0.727 / 0.81, n = 750).
- Weak in-distribution too: `tic_tac_toe-v1` test 0.523 / 1.14 (n = 472), `reasoning-control-v1` test 0.865.

## Jev Decision Index (HF Space `multimodalart/jev-decision-index`; read 2026-11-21 from its README + `data/methodology.json`, `data/index.json`)
UNVERIFIED as to our harness; VERIFIED as to what the Space publishes.
- Public suite = 37 benchmarks in 5 areas (Knowledge & Reasoning, Language Understanding, Retrieval & Classification, Tools & Automation, Arts & Human Taste): MMLU-style QA, GPQA, HellaSwag, WinoGrande, GSM8K, HLE, ANLI/NLI4CT/ContractNLI, BANKING77/CLINC150/ESCI, BFCL/API-Bank/ToolRet, ChessBench, CLadder, MuSR, ... Full score = 60% **private** same-skill tests + 40% private new-domain tasks; public benchmarks are shown but do not count. Edition 0.4.1 reports 110,201 requests (CLAUDE.md says 132,422: edition mismatch, recheck before submitting).
- Rules: no truncation, no option pruning, no prompt tuning; unsupported/unanswered counts as WRONG. Our limits (4,096 tokens, 16 options, English only) would cost points: e.g. CLINC150 has 151 options; a 2-24-option model (Tev1) leaves 12.3% of requests unanswered, and we cap lower at 16.
- Contamination is policed: entrants were removed or penalised for training on test items (e.g. a LoRA trained on 800 MMLU-Pro test items was removed).
- Small entrants (public_skill, params): GS1 2B 62.2, Valen 1.5 2B 61.9, Meteor-2B 61.0, GLiDE 2B 60.6, Meteor-0.8B 58.8, GS1 0.8B 56.1; top overall 75.3 (27-57B). Almost all are full fine-tunes of general LLMs trained on broad data. Our training distribution (12 synthetic Open-Jev generators) overlaps with only a few suite benchmarks (support-ticket/intent routing, tool-call yes/no, game-state choice such as ChessBench, spam/phishing), so expect a weak placement without a much broader training mixture.
- Candidate nearer to our data (not inspected): `reachjalil/jevlogs-log-triage-benchmark`, `reachjalil/jev-luna-pagerduty-trigger` (log triage / paging decisions, cf. our security_incidents and agent_trace_observability sources).

## Candidate extra corpus: `SargeDev/jev-distill-corpus-v3` (card + `train.jsonl` read 2026-11-21): NOT usable as published
VERIFIED from the card and the 655,806-row train file:
- **It contains our evaluation rows.** The `openjev_v2` stream (74,626 train rows) is a repackaging of Open-Jev `release-v2-redistributable`; matching ids, 14,336 of those rows are in OUR validation (2,863), calibration (3,613) or **test (7,860 of our 10,356 test rows)** splits. Its own `ood` split is Open-Jev's OOD. Any use must drop `source == "openjev_v2"` entirely (we already have Open-Jev train).
- The rest: `yuri_v3` (443,977 rows): one-to-two-sentence **templated** scenarios (median 254 chars, ~60 tokens, p99 303 chars) over 53 domains, labels **distilled from Jev 1.13 (a closed TypeSafe API) via OpenRouter**: soft targets are another model's opinions, and the licence/terms for redistributing API-distilled labels are UNVERIFIED despite the card's Apache-2.0. `yuri_v1` (137,203 rows): memory-relevance yes/no over QA passages (HotpotQA etc., includes Arabic text), labels from a 32B teacher.
- Neither stream is long authored policy text with ground-truth labels, so it would not fill the gap identified above. 0 of the 231 JevBench public states appear verbatim (prefix check only).
- Single-author dataset (50 likes). Not what "reputable" should mean here.

## Public human-labelled alternatives for "apply this rule text to a case" (licence per HF tags; formats and overlaps NOT yet inspected)
| corpus | what | licence tag |
|---|---|---|
| ShARC (`UCLNLP/sharc`) | rule text + scenario + question -> yes / no / irrelevant / more info | CC-BY-SA-3.0 |
| LegalBench (`nguha/legalbench`) | 162 legal reasoning tasks (rule application, classification); per-task licences vary | CC-BY-4.0 (check each task) |
| BoolQ (`google/boolq`) | passage -> yes/no | CC-BY-SA-3.0 |
| CUAD (`theatticusproject/cuad`) | contract clauses (span extraction; would need conversion) | CC-BY-4.0 |
| RuleTaker (`tasksource/ruletaker`) | synthetic rules + facts -> true/false | Apache-2.0 |
| FOLIO (`yale-nlp/FOLIO`) | logic over text, human-written | MIT, gated |
Avoid for training if we ever submit to the Decision Index (they are public benchmarks there): ContractNLI (also CC-BY-NC-SA), NLI4CT, ANLI, HellaSwag, WinoGrande, MuSR, GSM8K and others listed in the methodology.
