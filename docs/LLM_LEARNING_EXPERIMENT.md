# Local Qwen learning experiment

Implementation is available; a real Qwen3.5 SFT experiment has **not** been completed.
No benchmark scores in this document are model results.

The 2026-11-08 inspection found PyTorch 2.7.0+cu126, Transformers 4.57.0,
PEFT 0.18.1, datasets 4.5.0 and TRL 0.27.2. CUDA reports false and
`nvidia-smi` cannot communicate with its driver. RAM available was about
51 GiB. Importing Transformers causal-model classes fails with
`numpy.dtype size changed ... Expected 96 ... got 88`.
Qwen2.5 0.5B, 1.5B-Instruct and 3B-Instruct HF safetensors snapshots exist;
they are not Qwen3.5. The Ollama Qwen3.5 manifest directory exists, but GGUF
is not a trainable HF/PEFT checkpoint. No Ollama executable was found in PATH,
`/usr/local/lib/ollama`, `/opt/ollama`, `/usr/local/bin/ollama`,
`/usr/bin/ollama` or `/tmp/ollama`. Socket inventory was restricted by netlink
permissions; no inference-server availability is claimed.

The previously generated `/tmp/space-env-qualified-sequence-corpus-20261108-v1`
reported five eligible train episodes, all from scripted privileged policies.
It currently fails cold replay with `task runtime replay mismatch`. These
artifacts have not been repaired, relabeled, or used for real training.
Sources are under concurrent development: regenerate/verify corpora and execute
live runs only after the source revision is frozen.

## Execution

From `/home/user/workspace/space-env`, with the existing Python environment:

```bash
PYTHONPATH=src python examples/llm/learning_experiment.py preflight
PYTHONPATH=src python examples/llm/learning_experiment.py preflight --model /absolute/local/Qwen3.5-HF
PYTHONPATH=src python examples/llm/learning_experiment.py prepare --corpus /absolute/verified/corpus --evaluation-split development
PYTHONPATH=src python examples/llm/learning_experiment.py run --corpus /absolute/verified/corpus --model /absolute/local/Qwen3.5-HF --output /tmp/qwen-learning-new --steps 1 --max-tokens 8192 --output-tokens 256 --max-calls 64
python -m pytest tests/test_learning_experiment.py -q
```

The model directory must contain Qwen config, local tokenizer with a chat
template and EOS, and complete safetensors weights. All HF loads use
`local_files_only=True`, `trust_remote_code=False` and offline environment
settings. Nothing is installed or downloaded. Default execution refuses other
Qwen versions. `--allow-other-qwen` explicitly authorizes a separately labeled
experiment with another local Qwen model; it does not establish Qwen3.5 results.
Preflight reports import/file readiness, not a guarantee that allocation fits.
CPU float32 LoRA can still require substantial time and activation memory.

Output must be new and separate from immutable corpus/model inputs. A run
performs base greedy native rollouts, rank-8/alpha-16 PEFT LoRA on q_proj/v_proj
with assistant-only cross entropy and SGD, saves the adapter, then evaluates
the adapted model on the identical held-out initial checkpoints. Seed, budgets,
learning rate, device, dtype and protocol are recorded. Defaults are a smoke
experiment, not a validated training recipe. No early stopping or tuning uses
held-out outcomes. Test split evaluation is explicit; use development while
choosing training settings.

## Data and provenance contracts

`prepare(corpora, evaluation_split='development')` calls real `verify_corpus`
(native replay and exact exports), checks geometry split claims across **all**
corpora and rejects repeated semantic trajectories. Caller group labels cannot
hide geometry leakage. Only train entries marked native SFT eligible become
examples. There must be train examples and held-out tasks. One evaluation task
is retained per initial checkpoint, regardless of demonstration policy.

Each SFT example comes from the public causal message prefix before a native
event, using the existing `wire_context`/`decode_calls` `json_fixed_v1`
contracts. Targets are native actions encoded as JSON, not claims of model
generation. Omitted roles, empty role choices and explicit waits are native no-op equivalents.
Preparation fills omitted roles with `wait` in fixed targets and compares only
non-wait commands per role (while still checking cadence and tool semantics).
Multiple non-wait commands per role, duplicate commands, unknown roles or malformed
lists cannot be normalized into a single valid action; these decisions are
counted as unrepresentable and omitted.
The exact number is recorded. Origin labels (including scripted/privileged)
remain explicit. Checkpoints, reference plans and evaluation demonstrations
never enter the tokenizer/training rows. Representable finish actions are retained; observe is outside this transport.
All checkpoints must declare a 5-second decision cadence. Other step durations
are omitted and counted. Targets contain only tool/action fields; SDK-generated
revisions and nonces are excluded from semantic round-trip equality. Both native
rollout passes use the same 16,000,000-byte dialogue budget. Invalid or unsuccessful episodes never supply positive targets.

`encode_example(tokenizer, row, max_tokens)` builds the chat generation prefix,
appends target tokens and EOS, and masks the entire prefix with -100. Oversized
examples fail; prompts and targets are never silently truncated. Generation
uses exactly the same prompt builder and greedy budgets before and after.
A non-EOS token-budget stop or malformed response ends that native rollout
with a recorded provider error. There is no action repair or scripted fallback.
Infrastructure failures abort the run rather than create completed results.

## Module API and coordination

The standalone module is `space_env.llm.learning_experiment`:

- `model_manifest(directory)` hashes resolved local files, allowing HF cache symlinks.
- `feasibility(model=None, allow_other_qwen=False)` reports packages/import errors and model requirements.
- `prepare(corpora, evaluation_split=...)` returns corpora receipts, causal rows,
  deduplicated evaluation records, geometry split mapping and source hashes.
- `run(corpora, model_path, output, steps=1, max_tokens=8192,
  output_tokens=256, max_calls=64, seed=0, learning_rate=1e-5,
  evaluation_split='development', allow_other_qwen=False)` returns a sealed manifest.

`manifest.json` starts with status `incomplete` before evaluation/training.
Only a fully finished run writes `completed`. It binds all package source hashes,
corpus receipts, exact training-row digest, origins, settings, base model files,
changed LoRA adapter files, losses and base/after row receipts. Sources, original
weights and corpus receipts are rechecked before completion. A source race fails
closed. Seals provide integrity, not independent attestation of generation.

For the separately owned `paired_evaluation.py`, each
`base/NNNN.json` and `after/NNNN.json` is a sealed object containing `episode`
(the unmodified native `LLMEpisode.record()`), `trace`, and `provider_error`.
The episode can be passed to the native verifier; its geometry can be obtained
through the existing semantic contract. Summary rows bind checkpoint, episode,
artifact hashes and native metrics. Pairing requires the identical initial
checkpoint and matching recorded inference settings. The adapter manifest binds
to the exact base model manifest. Traces retain prompt/schema, raw generated text,
token counts and delivered calls. This module does not compute bootstrap CIs or
import the separately developed pairing API. Consumers should keep provider
failures in denominators and cluster by geometry rather than treating decisions
or multiple policies as independent samples.

Tests use explicitly synthetic transport/tokenizer fixtures and actual native
sessions. They are evidence of contracts and leakage defenses, never evidence
of real Qwen capability or improved performance.

## Offline audit of completed runs

```bash
PYTHONPATH=src python examples/llm/learning_experiment.py audit --output /tmp/qwen-learning-new
```

`audit(directory)` never imports a tokenizer/model runtime or resamples a model.
It verifies sealed completed manifests against current package sources and the
actual local base-model file inventory, reruns corpus verification and causal
preparation (including data hash, origins, geometry splits and selected held-out
split), checks adapter file hashes/configuration, and cold-replays every native
base/after episode. Both passes must match the exact held-out initial checkpoint,
public task, tools, group/split and configured call/dialogue budgets. All summary
metrics must equal replayed native metrics and all task pairs must be present.

For each trace attempt, the audit reconstructs the public `wire_context`, checks
its messages/schema, decodes raw semantic JSON using SDK revision/nonce rules,
and binds the delivered call exactly to its native event and call ID. It checks
accounting ranges, finite timing, causal trace order and final provider stops.
Context-budget attempts are now explicitly recorded even when generation never
starts. Incomplete generations and decoder failures carry a terminal `stop_error`;
`generation_complete` records the EOS test. Missing/extra attempts, mismatched
provider errors, unexplained early stops, and unbound experiment files fail.
Training and generation both disable thinking, and generation is greedy with
cache disabled; these settings and the evaluation split are now in the manifest.

This audit checks artifact consistency, **not authentic generation or training**.
A self-consistent fabricated/resealed run cannot be distinguished from a real run
without trusted provenance or independent model execution. Token counts and EOS
flags receive structural/budget checks; they are not independently retokenized or
regenerated. Losses are checked for finite nonnegative values and step counts;
the audit does not repeat gradient updates. Tests explicitly use scripted native
rollouts and dummy weight/adapter bytes, never real model-result claims. Old
artifacts missing these trace/setting fields must not be relabeled as audited.

The default `max_calls` is now 64, allowing a 180-second native horizon at the
fixed 5-second cadence (36 steps) plus completion/recovery calls. Preflight emits
its capability JSON and exits with status 1 when blockers are present; the example
entry point propagates that status. Completed audits report `llm_resampled=false`.

## Positive-decision data quality gate

Preparation now returns `decision_counts` with `wait`, `nonwait`, and `finish`
counts over accepted SFT rows. A non-wait decision is a joint step issuing at
least one actual command, regardless of how many other roles wait. Finish rows
are counted separately and cannot satisfy the positive-command gate. The same
counts are stored in completed manifests and independently recomputed by audit.
Data containing only waits/finish is rejected even if its native episode is
SFT eligible (for example a successful wait-only tail checkpoint after all
commands were already scheduled). CLI `prepare` prints these counts.

The regression test uses the actual three-role `sequence_lanes` reference-policy
teacher. It checks that sparse-role non-wait targets survive, all representable
teacher decisions are retained, and decoded fixed targets reproduce the exact
native mission state and public observations at every step. Journal/nonce hashes
may differ; command effects must not. The teacher remains labeled
`scripted_privileged`; these tests are not real model evaluations.
