# AI-assisted Wiki source review

The Wiki source reviewer compares mapped public sources with `docs/wiki` and
opens a draft pull request containing evidence-backed findings. It never edits
Wiki pages, advances repository checkpoints, merges a pull request, or publishes
the Wiki. A CODEOWNER must implement or reject each finding and approve the exact
final pull-request head.

This is intentionally an assistant, not an autonomous documentation authority.
Exact quotations, structured output, and a second model assessment reduce error;
they do not prove that a proposed statement is true.

## Staged workflow

1. The collector reads the verified public repositories in
   [`wiki-sources.json`](wiki-sources.json).
2. The configured Azure OpenAI deployment produces a bounded proposal.
3. A separate stateless request independently checks whether the supplied
   evidence semantically supports the complete proposal. Unsupported or
   uncertain proposals become manual-review findings.
4. The workflow opens a **draft**, findings-only pull request containing the
   comparisons, exact evidence, prompt version, resolved model identity, and
   proposed checkpoint revisions.
5. A human checks the full source comparison, edits `docs/wiki` if justified,
   updates only the checkpoints actually reviewed, and marks the PR ready.
6. The `Wiki AI human attestation` check passes only after a catch-all
   `CODEOWNERS` owner approves the current head SHA. A later push invalidates the
   check until the new head is approved. The read-only check runs from the
   trusted default-branch workflow and checks out only the base revision.
7. The ordinary protected Wiki workflow validates and, after merge, separately
   publishes the reviewed source.

The required status check and review requirement must also be configured in the
repository ruleset. Repository files can provide and verify the check, but cannot
make their own branch-protection settings mandatory.

## Public repository evidence

Each manifest entry maps a default branch and one or more Wiki pages. An optional
`review_paths` array defines owner-maintained Git path patterns; it defaults to
`["**"]`. Paths outside an explicit scope are recorded but their contents are not
sent to the model. Scope changes require the same owner review as Wiki changes.

For every in-scope incremental change the collector:

- verifies the configured branch is still the remote default;
- requires the recorded revision to be an ancestor of the new head;
- records every changed path;
- collects bounded UTF-8 patches and complete changed-file snapshots; and
- stops that source for binary, inaccessible, rewritten, non-UTF-8, or oversized
  in-scope evidence rather than silently omitting it.

The request is capped at 100 fragments and 256 KiB across evidence and mapped
Wiki pages. Exceeding a cap means manual review. Repository text is untrusted data
and cannot choose pages, instructions, or permissions.

An incremental comparison cannot find old Wiki errors when the upstream source
has not changed. The monthly schedule therefore performs a bounded full audit of
the current files selected by `review_paths`, including sources whose head SHA is
unchanged. Broad scopes may deliberately require manual review; owners can narrow
them only after confirming that the omitted paths cannot substantiate mapped
Wiki claims.

Automatic and PR-producing runs require `unverified_repositories` to be empty.
Manual dry runs still display unresolved entries so owners can resolve a moved,
renamed, private, or intentionally excluded source. An exclusion is a documented
owner decision, not an inference made by the model. Move a deliberately excluded
source to `excluded_repositories` with its mapped pages, reason, approving
`@CODEOWNER`, and UTC `approved_at` timestamp; exclusions remain visible in every
findings report.

## Model and evaluation controls

The Azure request uses Entra authentication and the Responses v1 API. Both model
calls use strict JSON schemas and `store=false`; the latter is required because
Responses are stored by default according to the
[official OpenAI documentation](https://developers.openai.com/api/docs/guides/migrate-to-responses).
No API key is used.

`AZURE_OPENAI_EXPECTED_MODEL` pins the resolved model identity returned by Azure.
The run fails if the deployment resolves to a different model. Every finding
records the deployment, resolved generator and verifier models, and the version
and SHA-256 digest of both system prompts.

Before repository evidence is reviewed, the workflow runs the labelled cases in
`tests/fixtures/wiki-review-evals.json`. Every case must produce the expected
disposition and required replacement content. The initial cases are a small
regression seed, not a statistical guarantee, and are marked
`approved_for_scheduling: false`. PR-producing and scheduled runs refuse to start
until CODEOWNERS have expanded the suite to at least ten anonymised historical
false-positive, false-negative, no-change, and manual-review cases and set that
field to `true`. The suite SHA-256 is recorded with its result.

Prompts, complete responses, source snapshots, and evidence bundles stay in the
runner's temporary directory. Do not log, commit, or upload them as artifacts.
The tracked proposal contains only public repository evidence already safe for a
public pull request. The read-only analysis job transfers only that public report
through a one-day immutable artifact to a separate, short write-scoped job that
opens the draft PR.

## Repository setup

Create a protected `wiki-ai-review` GitHub Environment and an Entra application
or user-assigned managed identity whose federated credential trusts this
repository and environment. Assign only the `Cognitive Services OpenAI User`
role, at the Azure OpenAI resource hosting the selected deployment; do not grant
deployment-management, key-management, resource-group, or subscription roles.

Configure these variables:

| Variable | Purpose |
| --- | --- |
| `AZURE_CLIENT_ID` | Federated workload identity client ID |
| `AZURE_TENANT_ID` | Microsoft Entra tenant ID |
| `AZURE_SUBSCRIPTION_ID` | Subscription containing the identity/resource |
| `AZURE_OPENAI_ENDPOINT` | Pathless HTTPS `*.openai.azure.com` or `*.services.ai.azure.com` endpoint |
| `AZURE_OPENAI_DEPLOYMENT` | Responses-compatible deployment name |
| `AZURE_OPENAI_EXPECTED_MODEL` | Exact model identity expected in API responses |
| `WIKI_AI_REVIEW_ENABLED` | Set to `true` only after the preflight below succeeds |

The repository must allow GitHub Actions to create pull requests. Add these
required checks to the branch ruleset:

- `Validate Wiki source`
- `Require current-head CODEOWNER approval`

Require at least one approving review and dismiss stale approvals. Do not allow
the automation identity to bypass the ruleset. GitHub currently creates
approval-required `pull_request` workflow runs for PRs opened by `GITHUB_TOKEN`;
a maintainer must approve those workflow runs before their checks execute.

Before enabling the schedule:

1. Expand and approve the labelled evaluation suite as described above.
2. Resolve every `unverified_repositories` entry or move it to the explicit
   owner-approved `excluded_repositories` object.
3. Review or narrow each repository's full-audit `review_paths`.
4. Run `Wiki source review` manually without opening a PR. Confirm the live eval,
   model identity, source collection, two-pass review, and offline tests pass.
5. Run it with `open_pr=true`. Implement the draft findings, approve the current
   head as a CODEOWNER, and confirm both required checks pass.
6. Merge or close the test normally. Set `WIKI_AI_REVIEW_ENABLED=true` only after
   the end-to-end review succeeds.

The weekly schedule reviews new revisions. The monthly schedule performs the
bounded full audit. The workflow refuses to compete with an existing automated
review PR.

## Commands

```bash
python scripts/review_wiki.py collect-repositories \
  --manifest docs/wiki-sources.json --wiki docs/wiki \
  --output /tmp/wiki-evidence.json

python scripts/review_wiki.py evaluate \
  --cases tests/fixtures/wiki-review-evals.json \
  --endpoint "$AZURE_OPENAI_ENDPOINT" \
  --deployment "$AZURE_OPENAI_DEPLOYMENT" \
  --expected-model "$AZURE_OPENAI_EXPECTED_MODEL" \
  --output /tmp/wiki-eval.json

python scripts/review_wiki.py review \
  --evidence /tmp/wiki-evidence.json \
  --endpoint "$AZURE_OPENAI_ENDPOINT" \
  --deployment "$AZURE_OPENAI_DEPLOYMENT" \
  --expected-model "$AZURE_OPENAI_EXPECTED_MODEL" \
  --output /tmp/wiki-review.json

python scripts/review_wiki.py report \
  --evidence /tmp/wiki-evidence.json --review /tmp/wiki-review.json \
  --evaluation /tmp/wiki-eval.json \
  --output docs/wiki-review-proposal.md
```

Add `--full-audit` to collect bounded current-state evidence. Add
`--require-complete-inventory` for any run allowed to open a PR.

## SharePoint publication projections

The public reviewer must never read an arbitrary confidential SharePoint file or
attempt to decide which parts are safe. Pattern matching cannot guarantee that
names, identifiers, operational details, or sensitive free text have been
removed.

Instead, a private Azure workload may plug in a **dedicated publication
projection** maintained for the public Wiki. Its Microsoft Graph identity must
use the application `Files.SelectedOperations.Selected` scope and receive an
explicit `read` assignment only on the configured projection, attestation, and
signature files—not on the original confidential document. Grant that assignment
through a separate administrator-controlled provisioning process; the runtime
identity must not be able to widen its own access. File IDs, paths, audit records,
and Graph checkpoints stay in private configuration.

Before the projection can reach Azure OpenAI:

1. The source owner approves the exact UTF-8 projection as public Wiki input.
2. An organisation-approved DLP process records `dlp_status: "passed"`.
3. The approval service signs a canonical attestation with Ed25519. The
   attestation binds the projection SHA-256, opaque source and revision IDs,
   opaque approval and DLP scan IDs, allowed Wiki pages, classification, approval
   policy, and a maximum 31-day validity window.
4. The private workload pins the trusted public-key SHA-256 independently of the
   downloaded files and runs:

```bash
python scripts/collect_wiki_projection.py \
  --projection /private/public-projection.md \
  --attestation /private/public-projection.attestation.json \
  --signature /private/public-projection.attestation.sig \
  --approval-public-key /private/trusted-approval-key.pem \
  --expected-key-sha256 "$WIKI_APPROVAL_KEY_SHA256" \
  --wiki docs/wiki \
  --output /private/wiki-evidence.json
```

The verifier fails closed for a changed projection, untrusted key, invalid or
expired signature, missing DLP attestation, invalid page mapping, oversized or
non-UTF-8 content, and common obvious sensitive patterns. These checks verify the
approval chain; they do not turn a confidential document into safe public text.
Only the already-approved projection is emitted. Source paths, SharePoint IDs,
signatures, and approval identities are omitted.

The private workload may then pass that evidence to `review`. It must keep the
evidence and response inside its authorised boundary until a human approves the
resulting public Wiki diff. The public scheduled workflow does not access
SharePoint.
