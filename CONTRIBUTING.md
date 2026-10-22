# Contributing to Merlin

Thanks for taking the time to contribute. Bug reports, ideas and pull requests are all welcome.
By participating you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Reporting bugs and suggesting features

- Search the [existing issues](../../issues) first.
- Use the **Bug report** or **Feature request** form when opening a new issue.
- Security problems do not belong in public issues; see [SECURITY.md](SECURITY.md).
- Never paste your OpenAI API key or the contents of your `.env` file into an issue.

## Development setup

You need Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

pip install -e ".[dev]"
# The Top-K + GPT method also needs FAISS:
# pip install -e ".[dev,faiss]"

ruff check .
pytest
```

The test suite is offline: it never calls OpenAI and needs no API key. Please keep it that way, and use a fake
client when you test code that talks to the API. See the [README](README.md#getting-started) for how to set up
your own `.env` file when you want to try the classifiers for real (those calls are billed by OpenAI).

## Making a change

1. Fork the repository and create a branch from `main`, for example `fix/empty-level-handling`.
2. Keep the change focused. Unrelated clean-ups belong in their own pull request.
3. Add or update tests for the behaviour you changed.
4. Make sure `ruff check .` and `pytest` pass; the same two commands run in CI.
5. Open a pull request and fill in the template.

## Commit messages

This project uses [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>: <short summary in the imperative mood>
```

Common types: `feat`, `fix`, `docs`, `test`, `ci`, `build`, `chore`, `style`, `refactor`.
Examples: `fix: handle categories with an empty level 3`, `docs: clarify the FAISS setup`.

## Style

Code is linted with [ruff](https://docs.astral.sh/ruff/); the configuration lives in `pyproject.toml`.
`.editorconfig` covers indentation and line endings. Please do not mix formatting-only changes into
functional pull requests.

## License

By contributing you agree that your contributions are licensed under the project's [MIT License](LICENSE).
