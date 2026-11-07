# Wiki maintenance

[`docs/wiki`](wiki) is the reviewed source for the
[`nhp_products` Wiki](https://github.com/The-Strategy-Unit/nhp_products/wiki).
GitHub renders that content from the separate `nhp_products.wiki.git`
repository.

## Update policy

- Use [`wiki-sources.json`](wiki-sources.json) to find the pages associated with
  each public repository and its last reviewed default-branch SHA.
- Compare that SHA with the repository's current default branch. A changed SHA
  triggers review, not an automatic content edit.
- Change only claims made inaccurate by reviewed commits. Preserve all other
  page text, especially private paths, permissions, and internal procedures
  that cannot be verified publicly.
- Keep existing filenames. Retire a page with a reviewed notice and replacement
  link rather than deleting it.
- Update the manifest SHA after reviewing a repository, even when no page edit
  is required. Add newly referenced public repositories to the manifest.

The initial import matches Wiki commit
`d22e900aeb2315313f6556a26599e7504dd03f59`. The manifest records the public
repository heads reviewed during the initial content audit. Never commit
credentials, environment values, or sensitive records.

## Review and publication

1. Edit `docs/wiki`, and update `wiki-sources.json` when applicable.
2. Open a pull request. The `Wiki` workflow validates the source and performs a
   preflight against a temporary Wiki clone without publishing.
3. Before the first merge, exercise the workflow in a disposable repository
   Wiki. Then protect the real `wiki` Environment with required reviewers and
   prevent self-review.
4. Merge the reviewed pull request to `main`, then run the workflow manually
   without selecting `publish` to preview the authenticated push.
5. Complete one reviewed manual publication. Then set the repository variable
   `WIKI_AUTO_PUBLISH=true` to queue future merges for protected publication.

The publisher uses the short-lived `GITHUB_TOKEN`, prevents concurrent pushes,
refuses to delete untracked Wiki pages, and never resets, rebases, or
force-pushes Wiki history. Manual workflow runs are dry runs unless `publish`
is explicitly selected. A dry run checks authentication and previews the
update; only a controlled publication proves write access end to end. Because
the workflow uses path filters, do not require its check on unrelated pull
requests.

Direct Wiki edits are not durable: the next publication restores `docs/wiki`.
The source and published locations are:

- `https://github.com/The-Strategy-Unit/nhp_products/tree/main/docs/wiki`
- `https://github.com/The-Strategy-Unit/nhp_products/wiki`

## Rollback

Restore the last known-good source in a new pull request:

```bash
git restore --source=<last-good-source-sha> -- docs/wiki docs/wiki-sources.json
```

Review and merge it normally. The workflow will publish a new restoration
commit, preserving both histories. Do not reset, rebase, force-push, or normally
use the Wiki UI to revert, because that would diverge from `docs/wiki`.

## Local checks

```bash
uv run python scripts/sync_wiki.py validate docs/wiki
uv run pytest tests/test_sync_wiki.py

git clone https://github.com/The-Strategy-Unit/nhp_products.wiki.git /tmp/nhp-products-wiki
uv run python scripts/sync_wiki.py check docs/wiki /tmp/nhp-products-wiki
```
