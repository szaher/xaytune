# Security and Governance

## 1. Agent authority

Agents may:

- inspect curated experiment context
- propose typed actions
- propose candidates
- explain rationale

Agents may not directly receive:

- Kubernetes credentials
- cloud credentials
- secret store tokens
- unrestricted shell
- unrestricted local filesystem
- raw runtime client objects

## 2. Execution authority

Only the controller executes approved actions.

Flow:

```text
Planner/Agent
  ↓
schema validation
  ↓
PolicyEngine
  ↓
BudgetLedger
  ↓
CapabilityResolver
  ↓
approval if needed
  ↓
Controller
  ↓
RuntimeBackend
```

## 3. Secrets

Secrets are runtime/backend configuration.

`TrainingExecutionSpec` should contain secret references, not secret values, when possible.

Example:

```python
SecretRef(
    provider="kubernetes",
    name="hf-token",
    key="token",
)
```

Core persistence should avoid storing resolved secret material.

## 4. Audit

Record:

- actor
- proposal
- policy decision
- approval
- execution
- result
- timestamps

## 5. Dependency execution

Trainer compilers may need custom code.

Execution policy should eventually control:

- allowed container registries
- image digests
- allowed package sources
- network access
- custom Python entrypoints

MVP can be permissive locally but schemas should support stricter platform policy.

## 6. Prompt injection / data poisoning

Experiment agents should not blindly consume arbitrary training examples or runtime logs as high-trust instructions.

Curated context should distinguish:

- untrusted data
- runtime evidence
- policy
- system instructions
- user objectives

## 7. Supply chain

Record:

- package versions
- container digest
- compiler plugin version
- runtime plugin version
- code revision

## 8. Human approval

Actions that may be configured to require approval:

- base model switch
- dataset switch
- external data access
- new runtime/provider
- increased GPU reservation
- budget override
- destructive artifact deletion
- export/publish
