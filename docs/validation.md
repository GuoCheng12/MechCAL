# Release Validation

The 0.1.0 release was checked using Python 3.11. The following test groups
passed in the release preparation environment:

| Test group | Passed | Scope |
| --- | ---: | --- |
| Core regression and packaging | 243 | Offline, with fake model responses |
| MCP integration | 2 | Local subprocess transport |
| Optional ChemGraph adapter | 13 | Adapter tests with ChemGraph installed |

The groups contain 258 distinct tests. The default test run skips the
optional ChemGraph module when its dependencies are absent and deselects
the two integration tests. The MCP checks require an environment that
permits local subprocess communication.

Additional checks passed:

- Python source compilation and the configured Ruff correctness checks.
- A synthetic, offline metric example.
- Wheel and source-distribution builds.
- Wheel installation into a separate environment, imported from outside the
  repository with Python isolated mode.
- Installed command help and bundled prompt and rubric availability.
- Source and distribution scans for credential patterns, known private
  endpoints, machine-specific paths, and excluded research artifacts.

Test environments reused locally installed dependencies. This is not a
verification of dependency resolution in a fully clean environment. The
GitHub Actions workflow performs a fresh installation when it runs.

No live model API experiment or Amesp scientific calculation was run for
this packaging check. These checks establish release packaging and covered
behavior, not reproduction of paper results. Secret-pattern scans also do
not replace a final content review before publication.
