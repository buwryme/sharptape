# Contributing to Sharptape

Thank you for your interest in contributing! We welcome bug reports, feature requests, and pull requests.

**Before opening a PR, please read this guide entirely.**

---

## Code Quality Requirements

### All Code Must Be Well-Researched

- **No unchecked code.** Every algorithmic or architectural change must be based on research, industry practice, or documented evidence.
- **Cite your sources.** Include links to papers, benchmarks, GitHub issues, or documentation that justify the change.
- **Explain the rationale.** In your PR description, explain *why* the change is needed and *what problem it solves*.

### Examples of Acceptable PRs

✅ "Fixed color space handling in ESRGAN by checking model scale support. Related: https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan/issues/80"

✅ "Optimized frame filtering from per-frame spawning to batch mode. Reduces overhead from O(n) process spawns to O(1). Benchmark: 3000 frames 5→50s on GTX 1650."

### Examples of Unacceptable PRs

❌ "Changed some settings because I think it's faster" (no citation, no benchmark)

❌ "Random code cleanup" (vague; must explain each change)

❌ "This might fix the black frames issue" (speculative; needs investigation + proof)

---

## Pull Request Workflow

1. **Fork the repo** and create a feature branch (`git checkout -b fix/my-feature`)
2. **Make your changes** and test thoroughly
3. **Write a clear PR description** that includes:
   - What problem does this solve?
   - How does it work?
   - Links to research, issues, or benchmarks
   - Test results if applicable
4. **Submit the PR** for review

### Maintainer Review

All PRs will be **thoroughly reviewed and tested** before merging. Expect:

- Code style and correctness checks
- Performance benchmarking (where applicable)
- GPU compatibility testing (if GPU-related)
- Verification that cited sources are legitimate
- Testing on multiple hardware configurations

**This is not personal.** We're protecting the project's reliability.

---

## Reporting Issues

### Bug Reports

Include:

- **Steps to reproduce** — exact commands and settings
- **Expected behavior** — what should happen
- **Actual behavior** — what actually happened
- **Debug logs** — run with `DEBUG=1` and paste relevant output
- **Hardware info** — GPU model, driver version, OS

### Feature Requests

Include:

- **Use case** — why is this needed?
- **Proposed solution** — how would you implement it?
- **Alternatives** — other ways to solve this problem

---

## Code Style

- **Python** — Follow PEP 8; use type hints where practical
- **Comments** — Document *why*, not *what*
- **Commits** — Use clear, atomic commit messages

---

## Questions?

Open an issue or ask in discussions. We're here to help.

**Thank you for contributing to Sharptape!** 🎬
