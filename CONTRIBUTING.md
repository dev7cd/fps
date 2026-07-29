# Contributing to FPS

Thanks for your interest in FPS (Fiber Packing System). This project is
maintained by a single author in an academic setting; issues and pull
requests are welcome, but please read this before opening one.

## Licensing of contributions

FPS is dual-licensed: [AGPL-3.0-or-later](LICENSE) for open use, with a
separate commercial license available on request (see [NOTICE](NOTICE)).
Offering a commercial license alongside the copyleft one only works if the
copyright holder can actually grant both — which requires holding (or having
a clear license to) the rights on every line of code in the repository.

By submitting a contribution (pull request, patch, or any other form of
proposed change), **you agree that**:

- You have the right to submit the contribution under these terms (it is
  your own original work, or you have the necessary rights to submit it).
- You license your contribution under the AGPL-3.0-or-later, **and** you
  grant the project maintainer the right to also relicense your contribution
  as part of the commercial-license offering described in [NOTICE](NOTICE).
- You retain copyright over your own contribution; you are not transferring
  ownership, only granting the licenses above.

If you are not able to agree to the second point (e.g. an employer or
institution claims rights over your contributions and won't allow
dual-licensing), please say so in the pull request — the maintainer may
still be able to accept the contribution under AGPL-only terms, but this
needs to be discussed explicitly rather than assumed.

For substantial contributions, adding yourself under a `Contributors`
heading in the README (as distinct from `Author`) is welcome and preserves
attribution regardless of the licensing terms above.

## Reporting bugs / requesting features

Please open a GitHub issue with:
- FPS version (`pip show fps` or the git commit hash)
- Python version and OS
- A minimal command or script reproducing the problem
- The full traceback, if any

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,gui]"
pytest
```

## Security issues

Do not open a public issue for a security-relevant bug — see
[SECURITY.md](SECURITY.md) instead.
