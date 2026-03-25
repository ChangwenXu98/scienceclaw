# Development Log: Superhydride Screening Pipeline

## Goal
Make the ScienceClaw agent **autonomously** screen superhydride candidates for superconductivity. The agent should reason about what structures to enumerate, relax them with UMA, and assess stability — without hardcoded pipelines.

---

## Phase 1: Skills Infrastructure (COMPLETED)

### What was implemented
- **DFT skill** (`skills/dft/`): Submit/poll/retrieve scripts for SLURM DFT jobs (DREAMS placeholder)
- **HPC skill** (`skills/hpc/`): SLURM reference guide (no scripts — teaches agent to use sbatch/squeue directly)
- **UMA skill** (`skills/uma/`): `uma_relax.py` — relax structures with Meta's UMA model via fairchem. Supports `--pressure`, `--relax-cell`, `--mp-id`, `--output-cif`
- **Artifact/reactor wiring**: Registered uma, hpc, dft, code-execution in `SKILL_DOMAIN_MAP` and `SKILL_INPUT_MAP`
- **Agent setup**: Added uma, hpc, code-execution to materials preset in `setup.py`; created MatSim agent

### Status: RESOLVED
All skills work standalone. `uma_relax.py` tested end-to-end: submitted SLURM job to venkvis-h100, relaxed Si (mp-149) in 59 seconds on H100.

---

## Phase 2: Standalone Screening Pipeline (COMPLETED but WRONG APPROACH)

### What was implemented
- `hydride_enumerate.py`: Hardcoded prototype structures (H3S, CaH6, YH9, LaH10, MH12) with Wyckoff positions, metal substitution
- `hydride_screen.py`: Full pipeline — enumerate → relax at 0/150 GPa → formation energy → convex hull
- Ran screening: 40 candidates × 2 pressures = 80 relaxations in 71 min on H100

### Issue identified
**This was a standalone script, NOT driven by the agent.** The agent framework (`scienceclaw-post`) was never involved. The results were correct scientifically, but the *process* violated the development goal: the agent should do this autonomously.

### Resolution
Deleted the hardcoded pipeline scripts. Skills should teach capabilities, not encode strategy. Updated CLAUDE.md with development philosophy.

---

## Phase 3: Agent Framework Integration (IN PROGRESS)

### Issue 3.1: Agent can't execute multi-step workflows
**Status: OPEN**

The `scienceclaw-post` skill executor calls each skill **once** with a single set of parameters. It cannot:
- Loop over multiple structures
- Chain outputs between skills (materials → uma)
- Write and execute code
- Submit SLURM jobs via bash

**Root cause:** Framework designed for information retrieval (pass query, get JSON), not computational workflows.

### Issue 3.2: Skill executor injects `--query <topic>` as fallback
**Status: RESOLVED**

When the LLM doesn't provide params, the executor falls back to `--query <entire topic string>`, which breaks scripts that don't accept `--query`.

**Fix applied:** Added special handling in `deep_investigation.py` to strip query params for `code-execution` skill (line ~328). Also added similar handling needed for `uma` (accepts `--structure`/`--mp-id`, not `--query`).

### Issue 3.3: Skill selector doesn't generate code in PARAMS
**Status: OPEN**

The skill selector LLM prompt asks for `PARAMS: {"key": "value"}` on a single line. Generating multi-line Python code inside a JSON string on one line is unreliable. The LLM either leaves params empty or generates broken snippets.

**Contributing factors:**
- `max_tokens=800` was too small (increased to 2000)
- Skill catalog only shows `name: description[:80]` — no parameter docs
- No examples of code-execution PARAMS in the prompt

**Partial fix applied:** Added computational workflow guidance to skill selector prompt. But LLM still doesn't generate code.

### Issue 3.4: 60-second timeout too short for computation
**Status: RESOLVED**

Skill executor had hardcoded `timeout=60` for all skills.

**Fix applied:** Added conditional timeout in `deep_investigation.py`: 300s for `code-execution`, `uma`, `dft`; 60s for everything else.

---

## Phase 4: Two-Track Solution (CURRENT)

### Track A: Pre-built screening script (compatible with existing framework)
Re-introduce a `hydride_screen.py` as a proper ScienceClaw skill script that the executor can call with simple parameters (e.g., `--metals La,Y,Ca --stoichiometries 6,10 --pressures 0,150`). This works within the existing one-call-per-skill design.

**Trade-off:** The agent doesn't reason about *what* to screen — it just calls the script. But the pipeline runs through `scienceclaw-post`.

### Track B: Two-stage code generation
Modify the investigation pipeline so the LLM first generates a **plan**, then a separate LLM call generates **executable Python code** for the `code-execution` skill. This separates "what to do" (planning) from "how to do it" (code generation).

**Goal:** Agent autonomously writes and executes the screening code.

---

## Phase 4 Results

### Track A: Pre-built `uma_screen.py`
- **Status: PARTIALLY WORKING**
- Script created and works standalone (dry-run verified)
- Agent correctly selects UMA skill and calls `uma_screen.py`
- Query injection fixed — `--query` no longer passed to computational skills
- GPU auto-detection added — falls back to CPU when no GPU available
- **Remaining issue:** UMA model loading on CPU exceeds 300s timeout. The agent runs on a login node without GPU.

### Track B: Two-stage code generation
- **Status: PARTIALLY WORKING — code generation succeeded!**
- LLM generated 9,546 chars of executable Python code autonomously
- The two-stage approach (plan → generate code → execute) works conceptually
- **Remaining issue:** Same timeout problem — generated code tried to load UMA on CPU
- **Key insight:** The code generation itself is the breakthrough. The agent CAN write computational code. The bottleneck is execution environment (no GPU on login node).

### Root Issue: Execution Environment (RESOLVED)
Both tracks initially failed because the agent runs on a login node without GPU. UMA requires GPU.

**Solution implemented:** Auto-SLURM submission. When no GPU is detected:
- Track A (`uma_screen.py`): Script detects `torch.cuda.is_available() == False`, writes a SLURM script, submits via `sbatch`, returns job info as JSON
- Track B (`code-execution`): The code generation prompt instructs the LLM to check for GPU and submit to SLURM if unavailable

### Final Test Results

**Track A: WORKING END-TO-END**
- Agent calls `uma_screen.py` → no GPU → auto-submits to SLURM → job runs on H100 → 20 relaxations + convex hull in 8.5 min
- All 10 candidates on the convex hull at 0 GPa
- Pipeline fully autonomous through `scienceclaw-post`

**Track B: PARTIALLY WORKING**
- Agent generates 10,247 chars of Python code autonomously (two-stage LLM call works)
- Code correctly attempts SLURM submission when no GPU detected
- SLURM job failed (exit code 2) — likely a script path or syntax issue in the LLM-generated SLURM script
- The code generation capability itself is proven; execution reliability needs improvement

## Phase 5: Removing Hardcoded Prototypes (IN PROGRESS)

### What was implemented
- **`structure-enumeration` skill**: Generic element substitution tool. Accepts --prototypes (formulas to fetch from MP), --metals, --output-dir. Not hydride-specific.
- **`uma_screen.py` rewritten**: Removed all hardcoded prototypes (PROTOTYPES dict). Now reads CIF files from --structures-dir (default: ~/.scienceclaw/enumerated_structures). The agent decides what to screen.
- **LLM parameter extraction**: When structure-enumeration gets no params from the LLM selector, a separate LLM call extracts --prototypes and --metals from the topic string.
- **Default chaining via filesystem**: enumeration writes to ~/.scienceclaw/enumerated_structures/, uma_screen reads from the same path by default.

### Standalone tests
- `enumerate_structures.py --prototypes LaH3,CaH2 --metals Y,Sc,Ce` works: fetches from MP, substitutes, writes CIFs
- `uma_screen.py --structures-dir ~/.scienceclaw/enumerated_structures --dry-run` works: finds CIFs, shows plan

### Agent framework tests
- Blocked by Anthropic API 529 (overloaded) errors — the LLM skill selector call fails, causing fallback to keyword-based selection that only picks `materials`
- The parameter extraction and SLURM auto-submit code was never reached
- Need to retry when API is available

### Key design: 3-skill pipeline
The agent's autonomous flow is now:
1. `materials`: search MP for prototype hydride structures (LLM decides which)
2. `structure-enumeration`: fetch prototypes from MP, substitute metals → CIFs to well-known dir
3. `uma` (uma_screen.py): read CIFs from well-known dir → relax → formation energy → hull

Each skill runs once. Chaining happens through the filesystem (shared directory). The LLM reasons about WHAT to screen; the skills handle HOW.

## Phase 5 Results (2026-03-25)

### Successful end-to-end autonomous run
The agent completed the full pipeline via `scienceclaw-post`:
1. `materials` ✓ — searched MP
2. `structure-enumeration` ✓ — LLM extracted {prototypes: LaH3,CaH2, metals: Y,Sc,Ce}, fetched from MP, generated 8 CIFs
3. `uma` ✓ — auto-submitted to SLURM (job 27018190), completed in 1m43s on H100

### Files changed in Phase 5
- `skills/structure-enumeration/SKILL.md` — NEW: generic enumeration skill docs
- `skills/structure-enumeration/scripts/enumerate_structures.py` — NEW: fetch from MP + substitute metals
- `skills/uma/scripts/uma_screen.py` — REWRITTEN: removed PROTOTYPES dict, reads CIFs from --structures-dir
- `skills/uma/SKILL.md` — UPDATED: documents new uma_screen.py interface
- `autonomous/deep_investigation.py` — MODIFIED: params.clear() for compute skills, LLM param extraction for structure-enumeration, default --structures-dir for uma
- `artifacts/artifact.py` — ADDED: structure-enumeration to SKILL_DOMAIN_MAP
- `artifacts/reactor.py` — ADDED: structure-enumeration to SKILL_INPUT_MAP

### Known issues
- **Agent doesn't read SLURM results:** The UMA skill submits to SLURM and returns job info, but doesn't wait for or retrieve results. Results sit in `uma_screen_output/slurm-*.out`.
- **Reasoning not recorded:** The LLM's parameter extraction reasoning (why it picked LaH3, CaH2) is printed to stderr but not persisted in journal or artifacts.
- **materials skill returns wrong data:** The materials skill returns ceramic screening by default (Si-C, B-C systems), not the hydride structures requested. It doesn't pass the topic as a query.
- **Untested:** Whether the agent can reason about "superhydrides" without specifying formulas.

## Phase 6: Remove Special-Casing, Add Generic Retry (2026-03-25)

### Problem
Lines 325-434 in `deep_investigation.py` contained skill-specific hacks:
- `params.clear()` for `('code-execution', 'uma', 'structure-enumeration')` only
- Hardcoded LLM prompt with hydride-specific hints ("Good choices for hydrides: LaH3...")
- Hardcoded `--structures-dir` default for `uma` only
- Hardcoded code generation specifically for `code-execution`

This violates the principle: no special processing for specific skills.

### Solution: Generic retry-with-SKILL.md
When ANY skill fails due to wrong parameters:
1. Read the skill's `SKILL.md` documentation
2. Send the error message + SKILL.md to an LLM call
3. LLM generates correct params based on the docs
4. Retry once with corrected params

This is fully generic — works for any skill without knowing its name.

### Files changed
- `autonomous/deep_investigation.py`:
  - REMOVED: all skill-specific param handling (lines 325-434)
  - REMOVED: hardcoded timeout for specific skill names
  - ADDED: generic timeout detection from SKILL.md content (GPU/SLURM keywords → 300s)
  - ADDED: generic retry mechanism after param errors — reads SKILL.md, asks LLM for correct params, retries once

### Test Results (2026-03-25)

**Prompt: "Screen superhydride candidates for superconductivity"** (no specific formulas)

| Skill | First attempt | Retry result |
|-------|--------------|--------------|
| materials | Timed out (60s) | N/A |
| structure-enumeration | Wrong params → failed | SKILL.md retry → `{prototypes: LaH10,CaH6,YH10,ScH3, metals: Y,Ca,Sc,Ce,Ba,La,Sr,Li,Na,K}` → **SUCCESS** |
| uma (uma_screen.py) | Wrong params → failed | SKILL.md retry → `{structures-dir: '.', pressures: 100,150,200}` → **SUCCESS** |

The agent autonomously decided prototypes and metals from just "superhydrides". No hardcoded hints.

### Remaining issues
- UMA retry picked `structures-dir: '.'` instead of `~/.scienceclaw/enumerated_structures` — works if CIFs happen to be there, but fragile. Need to pass prior skill output context to retry prompt.
- `materials` skill timed out (60s default for info-retrieval skills). Its SKILL.md doesn't mention GPU/SLURM so it gets the short timeout.

### Files changed
- `autonomous/deep_investigation.py`:
  - REMOVED: lines 325-434 (all skill-specific handling)
  - ADDED: generic retry mechanism (lines 379-441): on param error, read SKILL.md, ask LLM for correct params with script name context, strip markdown, extract JSON, retry once
  - CHANGED: timeout detection from hardcoded skill names to SKILL.md keyword scan

## Phase 7: Phonon Skill, Job Results, Multi-Step Screening (2026-03-25)

### What was implemented

**job-results skill** (`skills/job-results/`):
- `read_job_results.py`: reads SLURM output files, parses JSON results, filters by stability criteria, lists CIF paths for stable candidates
- Skips error/failed results when scanning output files
- Bridges gap between job submission and next pipeline step

**phonon skill** (`skills/phonon/`):
- `phonon_stability.py`: computes phonon properties via phonopy + UMA finite-displacement method
- Generates displaced supercells, computes forces with UMA, builds force constants
- Checks for imaginary modes → dynamic stability flag
- Computes thermal properties (free energy, entropy, heat capacity at 0-600K)
- Auto-submits to SLURM when no GPU available
- Reference: adapted from https://github.com/hyllios/utils/tree/main/benchmark_ph

### Files changed
- `skills/phonon/SKILL.md` — NEW: phonon skill documentation
- `skills/phonon/scripts/phonon_stability.py` — NEW: phonon calculation + stability check
- `skills/job-results/SKILL.md` — NEW: job results reading skill documentation
- `skills/job-results/scripts/read_job_results.py` — NEW: SLURM output parser + candidate filter
- `artifacts/artifact.py` — ADDED: phonon, job-results to SKILL_DOMAIN_MAP
- `artifacts/reactor.py` — ADDED: phonon, job-results to SKILL_INPUT_MAP

### Intended multi-step pipeline
1. `materials` — search MP for prototype structures
2. `structure-enumeration` — substitute metals, generate CIF candidates
3. `uma` (uma_screen.py) — relax all at 0/150 GPa, formation energy, hull analysis → SLURM
4. `job-results` — read screening results, identify stable candidates, list their CIF paths
5. `phonon` — compute phonon properties for stable candidates → SLURM
6. `job-results` — read phonon results, identify dynamically stable candidates

### Test Results (2026-03-25)

**5-skill pipeline test**: `scienceclaw-post --skills materials,structure-enumeration,uma,job-results,phonon`

| Skill | Status | Notes |
|-------|--------|-------|
| structure-enumeration | ✓ | Retry → prototypes: LaH10,CaH6; metals: Y,Ca,Sc,Ce,Ba |
| materials | ✓ | First-try success |
| uma | ✓ | Retry → submitted to SLURM, completed |
| phonon | ✓ (locally) | Retry → submitted to SLURM, but **SLURM job failed** |
| job-results | ✗ | Boolean flag bug (fixed: `--filter-stable True` → `--filter-stable`) |

**Phonon SLURM failure cause**: LLM passed `--structures-dir .` instead of the UMA output directory. The phonon skill doesn't know where UMA wrote relaxed CIFs because **skills don't share context**.

**Root limitation**: In a single `scienceclaw-post` call, skills execute sequentially but don't pass outputs between them. The phonon skill can't know the UMA output directory because it wasn't told.

**Mitigation options**:
1. Run in two sequential `scienceclaw-post` calls (step 1: enumerate+relax, step 2: read+phonon)
2. Use code-execution to write a chaining script
3. Enhance the framework to pass prior skill outputs to subsequent skills

### Files changed
- `core/skill_executor.py`: FIXED boolean flag handling (True → flag only, False → skip)
- `skills/job-results/scripts/read_job_results.py`: FIXED skip ERROR results when scanning

## Open Questions

1. Should the agent be able to call skills multiple times in one investigation? (Iterative execution)
2. How to pass outputs between skills? (State/context threading)
3. Should code-execution have access to GPU? (Currently runs on login node via subprocess)
4. How to handle long-running computations? (SLURM submission from code-execution?)
