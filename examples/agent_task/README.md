# Training a controller on the real mission runtime

Run from the checkout after installing the package:

```bash
python examples/agent_task/demo.py /tmp/task-demo nominal
python examples/agent_task/demo.py /tmp/task-reroute reroute
python examples/agent_task/demo.py /tmp/task-loss loss
python examples/agent_task/demo.py /tmp/task-ttl ttl
# Optional PettingZoo joint control:
pip install -e '.[marl]'
python examples/agent_task/demo.py /tmp/task-parallel nominal --parallel
```

Three spacecraft acquire two requests, process one onboard, relay to another
spacecraft, then deliver to a ground station. The catalogue is identical in all
variants. A scripted teacher supplies decisions; it is not a learned/expert policy.
Nominal, reroute and ACK-loss should fulfill 2/2 requests; TTL should fulfill 1/2.
All variants compare native SDK/Gym/tools/CLI-plan checkpoints. `--parallel` adds
PettingZoo parity, with independent role observations and a single mission clock.

The directory contains scenario/task/plan, mid-fragment and in-flight checkpoints,
the final episode and verified RL/SFT JSONL. Episode/checkpoint files are privileged
trainer artifacts. Never pass the whole directory to a policy as its observation.

```bash
space-env task-run /tmp/task-demo/scenario.json /tmp/task-demo/task.json /tmp/task-demo/plan.json /tmp/task-demo/cli-episode.json
space-env task-replay /tmp/task-demo/cli-episode.json
space-env task-export /tmp/task-demo/cli-episode.json /tmp/task-demo/cli-sft.jsonl --format sft
```

See [the contract](../../docs/AGENT_RUNTIME_CONTRACT.md) for masks, authority,
rewards, duration/discount semantics, seed policies and explicit accuracy limits.
