# Sources and selection — 2026-09-23

Candidates inspected directly from public repository contents:

- [K-Dense optimize-for-gpu](https://github.com/K-Dense-AI/claude-scientific-skills/blob/49c6e97775eaa18ba791bebe23162a70ae601c18/skills/optimize-for-gpu/SKILL.md): evidence-based GPU acceleration, but broad scientific-Python porting and library selection rather than a recurring ML training diagnosis workflow.
- [Orchestra ml-training-recipes](https://github.com/Orchestra-Research/AI-Research-SKILLs/blob/773a52944ba4747a18bd4ae9ade53fff041adcbc/10-optimization/ml-training-recipes/SKILL.md): broad training implementation recipes; not selected as the performance routine.
- [NVIDIA MoE optimization workflow](https://github.com/NVIDIA/skills/blob/0f72c29b9f9c65aab34a7faa72660dd28d9b995f/skills/nemo-mbridge-perf-moe-optimization-workflow/SKILL.md): specifically Megatron Bridge / MoE distributed training. Its scope does not match general dense-model training diagnosis.
- OpenAI curated skill tree at `49f948faa9258a0c61caceaf225e179651397431`: no dedicated training profiler entry found.

The general web-search endpoint failed with HTTP 404. Search therefore used GitHub repository tree APIs and fetched source files; this is a bounded search, not an exhaustive survey. This skill is independently written for the local workflow rather than an installed copy of a candidate.

Primary technical references, fetched during creation:

- [PyTorch profiler recipe source](https://github.com/pytorch/tutorials/blob/main/recipes_source/recipes/profiler_recipe.py): CPU/CUDA activities, scheduled warmup/active windows, record_function ranges and profiler overhead.
- [PyTorch performance tuning guide source](https://github.com/pytorch/tutorials/blob/main/recipes_source/recipes/tuning_guide.py): input pipelines, synchronization, host overhead, threading and NUMA.
- [NVIDIA System Management Interface](https://docs.nvidia.com/deploy/nvidia-smi/index.html): hardware telemetry and utilization semantics.

Verify version-specific profiler behavior against the installed framework. These sources do not establish the bottleneck of any particular run; the trace must do that.

Additional NVIDIA counter, temporal-monitoring and roofline guidance, checked 2026-09-24: see [gpu-efficiency.md](gpu-efficiency.md).
