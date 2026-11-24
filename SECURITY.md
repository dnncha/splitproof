# Security Policy

`splitproof verify` is built to inspect certificates written by other people, such as a
certificate committed alongside a paper or benchmark. `splitproof mcp` runs inside an AI
agent session whose arguments can come from untrusted reports, repositories, or datasets.
Both surfaces therefore treat certificates, policies, and tool arguments as untrusted input.

## Supported versions

| Version | Supported |
| --- | --- |
| `0.1.x` | Yes |

## Threat model for untrusted certificates and MCP calls

### Protections enforced by splitproof

1. **No arbitrary program execution from a certificate.**
   `splitproof verify` instantiates similarity backends from the policy embedded in the
   certificate. Every backend declares an explicit `policy_params` allowlist with types
   and numeric ranges (`src/splitproof/sim/__init__.py`). `construct()` rejects any
   parameter outside that allowlist. Parameters that name an executable or file path (such
   as MMseqs2's `binary`) are never settable from a policy or certificate. External
   processes (`mmseqs`) run via `subprocess.run` with an explicit argument list, never a
   shell.
2. **Working-directory path containment.**
   When `splitproof verify` is called without `--data`, it checks the `source.path`
   recorded inside the certificate and refuses any path that resolves outside the current
   working directory (`_recorded_source` in `src/splitproof/api.py`). This prevents a
   crafted certificate from reading arbitrary files elsewhere on the machine or verifying
   cleanly against a decoy file. Pass `--data` explicitly when the dataset lives outside
   the working directory.
   In `splitproof mcp`, every output path argument (`out`, `cert`, `report`) is resolved
   and confined to the current working directory (`_writable` in `src/splitproof/mcp.py`).
   An agent steered by untrusted content cannot overwrite files outside the working tree.
3. **Safe parsing and strict verification.**
   Policies are parsed with `yaml.safe_load` or `json.loads`; splitproof never uses
   `pickle` or `eval`. `verify` recomputes digests, unit structure, constraint
   measurements, and leakage diagnostics from scratch and checks JSON value types strictly
   (for example rejecting a boolean `true` substituted for an integer `1`). Policy-dropped
   rows (`__dropped__`) are accepted during verification only when the policy declares
   `on_span: drop` and the split's dropped rows match the exact set the policy drops.
4. **Memory bounds on quadratic inputs.**
   Similarity pair graphs can grow quadratically in the number of rows. Pair accumulation
   and LSH candidate generation are bounded by `--max-pairs` (default 20,000,000) and fail
   with `TOO_MANY_PAIRS` instead of allocating unbounded memory. `scheme.folds` is capped
   at 1,000 before data is loaded.

### What splitproof does not protect against

1. **Authorship and provenance.**
   A certificate is a reproducibility and integrity anchor, not a cryptographic signature.
   It proves that the recorded measurements re-derive from a specific policy, dataset
   digest, and split digest. It does not prove who generated the certificate or that the
   dataset itself was not fabricated. Sign `splitproof.cert.json` with GPG or Sigstore if
   you need authorship attestation.
2. **Untrusted `PATH` or local binaries.**
   When verifying or running a policy with `metric: identity`, splitproof resolves `mmseqs`
   from your `PATH`. Ensure your environment and MMseqs2 binary are trusted.
3. **CPU time on large or adversarial inputs.**
   Memory is bounded by `--max-pairs`, but recomputing MMseqs2 alignments or exact Tanimoto
   similarities on tens of thousands of rows can take minutes of CPU time (for example,
   20,429 human proteins took 20 minutes on one core under MMseqs2). If you run
   `splitproof verify` on untrusted PRs in shared CI, enforce job timeouts and standard
   container isolation.
4. **Lax policies.**
   A certificate built from a weak threshold (such as `kmer_jaccard` at `0.95`) will
   re-verify cleanly and report `satisfied` on that declared threshold. Always review the
   embedded policy and the nearest-neighbour diagnostics, not only the word `verified`.

## Reporting a vulnerability

Please do not open a public GitHub issue for security vulnerabilities.

Use GitHub's private vulnerability reporting on the repository
(`https://github.com/dnncha/splitproof/security/advisories/new`), or email
`dnncha@gmail.com` with a minimal reproduction (policy, dataset sample, or crafted
certificate) and the `splitproof --version` output.
