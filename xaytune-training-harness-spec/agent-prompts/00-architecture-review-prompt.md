# Coding Agent Prompt — Architecture Review

You are working in the Xaytune repository.

Before making code changes:

1. Read:
   - README.md from this spec package
   - 01-product-and-scope.md
   - 02-architecture.md
   - all ADRs
   - 17-coding-agent-contract.md
2. Inspect the current Xaytune repository.
3. Identify which existing modules can be reused.
4. Produce a short implementation note containing:
   - target files
   - compatibility risk
   - tests to add
   - architecture invariant checks
5. Do not modify code until this review is complete.

Hard rules:

- no repository rewrite
- no remote execution inside TrainerCompiler
- no direct state assignments
- no heavy imports in xaytune.core
- no scientific spec mutation in-place
- no LLM direct execution
