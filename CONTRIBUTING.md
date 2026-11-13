# Contributing to Monolingual

Thanks for improving Monolingual. This guide covers local development, the test
gate, and the automated release process.

## Development setup

The plugin is a directory loaded by Hermes Agent. Python 3.11 or newer is
required.

```bash
git clone https://github.com/gwyntel/monolingual.git
cd monolingual

python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r tests/requirements-ci.txt -r tests/requirements-lingua.txt
```

`lingua-language-detector` is optional at runtime because the plugin falls back
to Unicode script detection. It is required for the full test path.

## Running tests

Run pytest from the `tests/` directory. The repository root contains the
plugin's own `__init__.py`, so invoking `pytest tests` from the root can confuse
pytest's package discovery.

```bash
cd tests
python -m pytest . -q
```

Also run the no-lingua path, because detector fallback is part of the supported
behavior:

```bash
python3 -m venv ../.venv-no-lingua
../.venv-no-lingua/bin/pip install -r ../tests/requirements-ci.txt
../.venv-no-lingua/bin/python -m pytest . -q
```

Pull requests must pass the complete GitHub Actions matrix:

- Python 3.11, 3.12, 3.13, and 3.14 with lingua installed.
- Python 3.13 without lingua, exercising the Unicode fallback.
- Release build and tag-validation tests.

Keep changes fail-open. A detector or translation error must return the original
response instead of blocking the agent's output.

## Submitting changes

1. Create a topic branch, for example `fix/unicode-mixing`.
2. Make the smallest change that addresses the issue.
3. Add or update regression tests.
4. Run both local test modes above.
5. Open a pull request and wait for all CI checks to pass.
6. Merge only after review approval.

## Versioning

`plugin.yaml` is the single source of truth for the release version. Use
semantic versioning:

- **Patch** for fixes that preserve behavior and configuration.
- **Minor** for backward-compatible features or configuration additions.
- **Major** for breaking changes.

Do not edit version numbers in generated archives or GitHub Releases. Update
`version` in `plugin.yaml`, commit it, and create the matching tag.

A version with a pre-release suffix, such as `1.0.0-rc.1`, is published as a
GitHub pre-release. Stable versions are published as normal releases.

## Release process

Releases are tag-driven and test-gated. A tag does not publish directly; the
workflow first validates the tag, then runs the complete test matrix. The
publishing job declares `needs: [precheck, tests]`, so if any job fails, skips,
or is cancelled, no release is created.

1. Update `version` in `plugin.yaml`.
2. Update documentation if behavior or configuration changed.
3. Commit and merge the version bump to `main`.
4. Create and push an annotated tag that exactly matches the manifest version:

   ```bash
   git checkout main
   git pull --ff-only origin main
   git tag -a v0.1.2 -m "Monolingual v0.1.2"
   git push origin v0.1.2
   ```

5. GitHub Actions runs `.github/workflows/release.yml`:
   - Validates that the tag is exactly `v<plugin.yaml version>` before running
     the expensive matrix.
   - Runs the reusable test workflow in `.github/workflows/tests.yml`.
   - Verifies the tagged commit is reachable from `main`.
   - Builds `dist/monolingual-<version>.zip` from the runtime plugin files.
   - Writes `dist/SHA256SUMS`.
   - Creates a draft GitHub Release, uploads both artifacts, then publishes it
     with generated release notes.
6. Verify the published release:

   ```bash
   gh release view v0.1.2 --repo gwyntel/monolingual
   gh release download v0.1.2 --repo gwyntel/monolingual --dir /tmp/monolingual-release
   ```

### Local release dry run

The release builder is the same script CI uses. It refuses a tag that does not
match `plugin.yaml` and never creates an archive when validation fails.

```bash
python scripts/release.py --tag v0.1.2 --output dist
shasum -a 256 dist/monolingual-0.1.2.zip  # macOS; use sha256sum on Linux
```

### Failed releases

If tests fail, fix the problem on `main` and publish a new version and tag. Do
not move or reuse a published tag. If the workflow failed before publication,
its draft release is safe to retry: the next run replaces the draft's assets and
only publishes after all tests pass. If the workflow failed before creating the
GitHub Release and you must replace the tag, bump the patch version instead so
the source of truth and artifact name remain unambiguous.

Releases require the `contents: write` permission on the release job. Pull
request workflows have read-only contents access.
