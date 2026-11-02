# A versioned mission/data factory

```bash
python examples/factory/demo.py /tmp/factory-corpus
space-env factory-verify /tmp/factory-corpus
```

The demo uses the published native fixed-GCRS example, not a toy replacement engine.
Six geometry groups form train/holdout/OOD strata. Two draws per template and three
policies produce 36 evaluated episodes, including an unreachable closed station.
Duplicate policy/seed trajectories are removed; negative cases remain in RL/audit
and are excluded from SFT. Default held-out teacher gates are required. The small
deterministic corpus currently yields 48 RL and 20 SFT rows, not expert qualification.

For a coupled EO → CPU → ISL → ground corpus:

```bash
python examples/factory/demo.py /tmp/factory-eo --eo
```

This single 20-second engineering example explicitly opts out of held-out teacher
qualification; rows remain synthetic, non-expert demonstrations. It exercises the
native spacecraft/RF/DTN/thermal models and ledger/replay gates. It does not establish
production data diversity or fleet throughput.

The sibling `.spec.json` is the portable input. The same workflow can run via CLI:

```bash
space-env factory-schema /tmp/factory-schema.json
space-env factory-compile /tmp/factory-corpus.spec.json /tmp/factory-catalogue.json
space-env factory-build /tmp/factory-corpus.spec.json /tmp/another-corpus
space-env factory-verify /tmp/another-corpus
```

Destinations must not already exist. Check `bundle.json` status and `report.json`;
quarantine returns exit 1. Train only on the explicit `data/<split>` files and proper
observation/messages fields. Scenarios, seeds, benchmark and episode audit are
privileged trainer data. [Contract](../../docs/FACTORY_CONTRACT.md).
