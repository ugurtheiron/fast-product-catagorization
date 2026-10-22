# Security Policy

## Supported versions

Merlin is pre-1.0. Security fixes are made on the latest `main` branch and released as a new `0.x` version
when one is warranted. Older versions are not patched.

## Reporting a vulnerability

Please report vulnerabilities **privately**, not in a public issue or pull request.

1. Open the repository's **Security** tab and choose **Report a vulnerability** (GitHub private vulnerability
   reporting).
2. Include what you found, the affected file or function, steps to reproduce, and the impact you expect.

If the **Report a vulnerability** button is not available, open a public issue that only says you need a private
channel to report a security problem. Do not include any details in that issue; the maintainer will reply
with a way to continue privately.

This is a personal open source project maintained on a best-effort basis. You can expect an acknowledgement
within about a week and a fix or a clear answer as soon as the maintainer is able. Please give a reasonable
time to fix the problem before disclosing it publicly.

## What is in scope

- Code in this repository that mishandles input, files or credentials (for example unsafe loading of the
  embedding cache or the FAISS index).
- Anything that could leak the `OPENAI_API_KEY` from the `.env` file.

Vulnerabilities in third-party packages (OpenAI SDK, pandas, FAISS and so on) should be reported to those
projects.

## If you leaked an API key

If you accidentally committed an OpenAI key, revoke it in the OpenAI dashboard first and create a new one;
removing the commit is not enough.
