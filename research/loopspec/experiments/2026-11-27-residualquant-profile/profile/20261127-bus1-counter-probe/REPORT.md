# Counter permission probe

CUDA launch succeeded, but root NCU collection failed with ERR_NVGPUCTRPERM. No occupancy, cache, bandwidth or stall metric was measured. Driver535 and the effective root capability set exclude CAP_SYS_ADMIN. NVIDIA documents host enablement or container SYS_ADMIN as the prerequisite: https://developer.nvidia.com/ERR_NVGPUCTRPERM . No driver reload or global permission changes were attempted.

Evidence: analysis/probe.log and analysis/environment.json. Continue ordinary CUDA-event timing and explicitly perturbed clock64 stage measurements; these do not substitute for hardware-counter attribution.
