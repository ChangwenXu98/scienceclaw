# Development Log: Autonomous Superhydride Screening with ScienceClaw

## What We Built

We extended ScienceClaw — a framework for autonomous scientific investigation — to perform computational materials screening entirely autonomously. Given only the prompt **"Screen superhydride candidates for superconductivity"**, the agent writes code, submits GPU jobs, reads results, debugs errors, and iterates until the screening is complete. No human writes any pipeline code.

---

## Key Innovation: The Agentic Computation Loop

The central contribution is an **agentic computation loop** that replaces traditional script-based skill execution. Instead of calling a pre-built script once, the agent enters an interactive loop where it can:

- **Write files** — Python scripts, SLURM submission scripts, analysis code
- **Run commands** — `sbatch`, `squeue`, `python3`, `cat`, `ls`
- **Read output** — job results, error logs, generated data
- **Decide what to do next** — based on what it observes

This mirrors how a human researcher works at a terminal: write code, run it, check if it worked, fix errors, run again.

### Implementation

The loop lives in `autonomous/deep_investigation.py` (`_run_agentic_computation` method). When the `code-execution` skill is selected, the framework enters this loop instead of calling a subprocess. The LLM receives:

1. The research task description
2. All available skill documentation (UMA API, HPC cluster info, etc.)
3. Results from prior skills (materials search, etc.)

At each step, the LLM responds with a single JSON action. The framework executes it, feeds the result back, and the LLM decides the next action. Up to 50 steps are allowed.

### What the Skill Documentation Teaches

The agent learns computational APIs from SKILL.md files, not from its training data:

- **UMA skill** (`skills/uma/SKILL.md`): Documents the Python API for structure relaxation with `pretrained_mlip.get_predict_unit()` and `FAIRChemCalculator`, phonon calculation via `phonopy`, structure generation with `pymatgen`, and SLURM self-submission patterns
- **HPC skill** (`skills/hpc/SKILL.md`): Documents Artemis cluster partitions (`venkvis-h100`, `venkvis-a100`), SLURM directives, job submission/monitoring commands, and interactive GPU sessions

---

## Demonstration: Autonomous Superhydride Screening

### Task
"Screen superhydride candidates for superconductivity: build LaH10 and CaH6 structures with Y, Ca, Sc, Ce substitutions, relax with UMA at 150 GPa, compute phonon stability."

### What the Agent Did (50 steps)

**Phase 1: Planning and Structure Generation (Steps 3-18)**

The agent began by writing three scripts — for structure generation, UMA relaxation, and phonon analysis — and a SLURM submission script:

```
Step 3:  write_file  superhydride_screening.py     ← Structure generation code
Step 4:  write_file  uma_relaxation.py              ← UMA relaxation code
Step 5:  write_file  phonon_analysis.py             ← Phonon analysis code
Step 6:  write_file  run_superhydride_screening.sh  ← SLURM script
```

The generated `superhydride_screening.py` builds LaH10 (Fm-3m, SG 225) and CaH6 (Im-3m, SG 229) prototype structures from crystallographic Wyckoff positions using pymatgen, then substitutes the metal site with Y, Ca, Sc, and Ce to produce 8 candidate structures.

**Phase 2: First Submission and Self-Correction (Steps 7-10)**

The first SLURM submission failed because the agent used an incorrect partition name:

```
Step 7:  run_command  sbatch run_superhydride_screening.sh  ✗ (rc=1)
```

The agent then **checked the cluster configuration** and rewrote the script:

```
Step 8:  run_command  sinfo -o '%P ...' | head -20  ✓    ← Queried real partitions
Step 9:  write_file   run_superhydride_screening.sh        ← Rewrote with correct partition
Step 10: run_command  sbatch run_superhydride_screening.sh  ✓  (job 27020559)
```

This self-correction — checking `sinfo` to learn the correct partition names, then rewriting the script — is exactly what a human researcher would do when encountering an unfamiliar cluster.

**Phase 3: UMA Relaxation with Iterative Debugging (Steps 20-40)**

After reading the structure generation results, the agent wrote a UMA relaxation script and submitted it. The first version failed with an `ImportError` (wrong ASE API). The agent's response:

```
Step 22: run_command  sbatch run_uma_relaxation.sh  ✓      ← Submitted
Step 24: run_command  cat uma_relax_*.out           ✓      ← Read output: errors found
Step 27: write_file   uma_relaxation_fixed.py               ← Wrote corrected version
Step 29: run_command  sbatch run_uma_relaxation_fixed.sh ✓  ← Resubmitted
Step 31: run_command  cat uma_relax_fixed_*.out     ✓      ← Still errors
Step 32: write_file   check_ase_constraints.py              ← Wrote DIAGNOSTIC script!
Step 33: run_command  python3 check_ase_constraints.py  ✓   ← Tested which APIs exist
Step 35: write_file   uma_relaxation_final.py               ← Wrote final version
Step 37: run_command  sbatch run_uma_relaxation_final.sh ✓  ← Submitted
Step 40: run_command  cat ...uma_relaxation_150GPa_results.json  ✓  ← Success!
```

The remarkable step here is **step 32**: when the second fix still failed, the agent **wrote a diagnostic test script** to check which ASE modules are actually available in the environment. It used the test results to write a third version that finally worked. This three-round self-correction with diagnostics is sophisticated problem-solving behavior.

**Phase 4: Phonon Stability Analysis (Steps 42-49)**

With relaxed structures in hand, the agent wrote a phonon analysis script:

```
Step 42: write_file   phonon_stability_analysis.py     ← Full phonopy analysis
Step 43: write_file   run_phonon_analysis.sh           ← SLURM script
Step 44: run_command  sbatch run_phonon_analysis.sh ✓   ← Submitted
Step 46: run_command  cat phonon_analysis_*.out     ✓   ← Failed (phonopy issue)
Step 47: write_file   simple_phonon_analysis.py         ← Simplified version
Step 48: run_command  python3 simple_phonon_analysis.py ✓  ← Ran locally
Step 49: read_file    phonon_stability_results.json     ← Read results
```

When the full phonopy calculation failed on GPU, the agent wrote a simplified version that ran locally — adapting its approach based on what worked.

### Results

The agent produced:
- 8 candidate structures (LaH10, CaH6 types × 4 metals)
- UMA relaxation results at 150 GPa for all candidates
- Phonon stability analysis for all candidates
- All through 6 SLURM job submissions with 3 rounds of self-correction

### Known Limitations

The scientific results from this demo have inaccuracies (pressure unit conversion errors led to extreme volume compression). This is because the LLM's generated code sometimes uses approximations from its training data rather than the exact conversion factors in the SKILL.md documentation. However, **the workflow itself — the ability to autonomously write code, submit jobs, debug errors, and iterate — is the contribution**, not the specific numerical results.

---

## Architecture Overview

### Two Approaches (Track A vs Track B)

We developed and tested two approaches to autonomous computational screening:

**Track A: Pre-built skill scripts** (`dft` branch)
- Each pipeline step is a separate skill script (enumerate_structures.py, uma_screen.py, phonon_stability.py)
- The agent selects which scripts to call; the scripts handle the computation
- When scripts fail with wrong parameters, a generic retry mechanism reads the skill's documentation and asks the LLM to generate correct parameters
- Works reliably but encodes domain strategy in the scripts

**Track B: Agentic computation loop** (`track-b-code-execution` branch)
- A single `code-execution` capability where the agent writes all code itself
- The agent iteratively writes files, runs commands, reads results, and self-corrects
- No pre-built pipeline code — the agent reasons about what to do from skill documentation
- More flexible but less reliable on API details

### Key Framework Changes

| File | Change |
|------|--------|
| `autonomous/deep_investigation.py` | Added `_run_agentic_computation()` — the interactive LLM loop with write_file/run_command/read_file actions |
| `autonomous/deep_investigation.py` | Added generic retry-with-SKILL.md mechanism — on parameter errors, reads skill docs and asks LLM for corrections |
| `autonomous/deep_investigation.py` | Added prior skill output context passing — each skill's output is available to subsequent skills |
| `core/skill_executor.py` | Fixed boolean flag handling, None value skipping, leading dash stripping |
| `core/skill_selector.py` | Increased max_tokens for computational workflow parameter generation |
| `skills/uma/SKILL.md` | Comprehensive Python API documentation for relaxation, phonon, structure generation, SLURM patterns, verification checklist |
| `skills/hpc/SKILL.md` | Cluster reference guide with partitions, SLURM commands, interactive GPU sessions |
| `skills/code-execution/SKILL.md` | Documents the agentic loop actions (write_file, run_command, read_file, done) |

---

## What This Demonstrates

1. **An LLM agent can autonomously execute multi-step computational materials science workflows** — from structure generation through property calculation, with no human-written pipeline code.

2. **Self-correction is essential and effective.** The agent encountered and resolved: wrong SLURM partition names (checked `sinfo`), wrong Python API calls (wrote diagnostic scripts), failed SLURM jobs (read error logs, rewrote code). Three rounds of self-correction for a single task is typical.

3. **Skill documentation is the teaching mechanism.** The agent learns APIs from SKILL.md files, not from hardcoded logic. New computational capabilities can be added by writing documentation, not by writing pipeline code.

4. **The agentic loop pattern generalizes.** The same write/run/read/iterate loop could drive any computational workflow — not just superhydride screening. DFT calculations, machine learning training, data analysis pipelines — any task a researcher does at a terminal.
