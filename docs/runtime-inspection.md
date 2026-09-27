# Inspect research runtime

Start with the user's concrete question and timestamp the observation. Collect only the relevant read-only measurements. Useful starting commands are `nvidia-smi`, `free -h`, `df -h` on the exact target mounts, `ps -eo pid,comm,pcpu,pmem --sort=-pcpu`, and filtered `systemctl list-units`.

Map an active PID to its executable, working directory and cgroup before attributing it to a project. `/proc/PID/exe`, `/proc/PID/cwd` and `/proc/PID/cgroup` avoid dumping environment variables or credential-bearing command lines. A running service is not proof of successful work: consult a narrow status artifact or recent progress counters when needed.

Utilization is an instantaneous observation, not throughput, thermal safety or training convergence. For speed questions compare progress over a stated interval and account for validation/checkpoint phases. For temperature questions read actual GPU/sensor measurements and device-reported limits; do not infer healthy temperatures from utilization alone.

Do not stop jobs, change fans, clocks, affinity, cgroups, drivers or service configuration as part of inspection. If a fix is requested, follow its authorized scope. Before substantial writes check the actual destination filesystem, including cache and temporary locations.

Report what is running, measured resource use, uncertainty and the answer to the user's question. Avoid disclosing process arguments or unrelated logs that could contain credentials or clinical identifiers.
