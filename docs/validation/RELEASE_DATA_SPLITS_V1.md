# Explicit benchmark training splits

The native mission bundle contract derives `train/holdout` from a lineage hash.
B12 defines explicit geometry groups `train/development/test`. These authorities
are different: the frozen `test-orbits-b` group hashes to legacy `train`.
Native bundles remain immutable replay evidence and evaluation artifacts. **Do
not train from their raw benchmark JSONL labels.** This is an integration finding,
not a newly independent test case or a reason to relabel old native bundles.

`scripts/export_release_data.py` publishes a separate versioned dataset. It takes
split authority from the verified frozen benchmark cases, preserves the old field
as `legacy_lineage_split`, records the original row SHA and source report SHA, and
uses `space-env.release-rl/1` or `space-env.release-sft/1`. Original observations,
actions, rewards and native episode bindings remain intact. Benchmark case/group/
policy and provenance are metadata, not new agent observation features.

RL includes every verified row, including unsuccessful missions, in its correct
partition. SFT is emitted **only for the training partition** and only after the
suite-level teacher gates pass. Native per-case success alone cannot promote a
teacher. Development/test demonstrations are never silently placed in train.
The current suite fails teacher gates, so its approved export contains RL only.

Source bundle inventory and case bindings are checked first. Each source JSONL
is then hashed while streaming; changes cause atomic publication to fail. Output
inventory, row hashes, exact regenerated labels and source bindings are checked
again by the verifier. The default path reuses native evidence from the controlled
benchmark producer and performs consistency verification. `--native` additionally
replays source evidence; hashes alone do not authenticate a malicious producer.
The manifest explicitly records whether native replay was repeated at export.

```bash
python -I scripts/export_release_data.py export /tmp/release4g /tmp/release-training
python -I scripts/export_release_data.py verify /tmp/release4g /tmp/release-training
```

Use absolute script paths outside checkout. Add `--native` to export or verify
when an additional native audit is required. Existing outputs are not overwritten.
An incomplete source with failed worker rows is rejected, rather than dropping
those cases from the training denominator. The1GiB experiment is therefore not
an eligible complete source; the4GiB experiment has complete native evidence.
Its three strict64 MB storage-gate failures do not change the already verified
physical trajectories. They remain explicit resource findings.

The exporter streams at most10000 rows per partition/file and2GiB of JSONL in
this profile. It does not advertise an arbitrary-size data lake, universal dataset
schema compatibility, expert demonstrations or a protected multi-tenant service.
Training code must recognize the new schema and select the intended partition.
