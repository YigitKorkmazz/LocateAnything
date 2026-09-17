"""Production-grade high-throughput GRPO backend for LocateAnything-3B.

New backend, kept out of the existing `rl/` package on purpose. It imports
`rl.*` (parsers, rewards, MedCLIP, low-level PBD/NTP decode primitives) as a
fixed scientific reference and never modifies those modules. See DESIGN.md
for the architecture and README.md for the final report.
"""
