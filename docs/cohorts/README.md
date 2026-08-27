# The cohorts the production maps were built from

One UniProt accession per line, sorted. These record **which proteins each
published map covers**, so a reader can see what is in it and diff a later run
against it.

| file | accessions | structures AlphaFold served on 2026-08-26 |
|---|---|---|
| `chymo_A1_n367.txt` | 367 | 367 |
| `actin_B_n308.txt` | 308 | 308 |
| `chymo_full_n2703.txt` | 2,703 | 2,656 |
| `actin_full_n2784.txt` | 2,784 | 2,689 |

## These lists do not make the maps reproducible, and that is measured

Re-staging them on 2026-08-26 produced genuine 404s from AlphaFold for **142 of
5,487 accessions (2.6%)**, after trying both the v6 and v4 endpoints. Those
proteins were in the published maps, so AlphaFold served them earlier and does
not now. The two `*.no_alphafold_model.txt` files list exactly which.

The inputs are a live third-party service, not a fixed corpus. A rebuild next
month gets a different cohort again, and every number on the map moves with it.
**Treat the published maps as evidence produced by this pipeline on a dated
cohort, not as a reproducible artifact.**

What *is* reproducible is the method: `demo/multispace/` runs the entire DAG
offline from eleven committed structures in about two minutes, and CI runs it on
every push.

## Where the counts come from, since two are in play per cohort

The full cohorts carry two protein counts and both are correct:

* **2,703 / 2,784** — every accession in the cohort. The `physicochemistry` and
  `families` maps are built from UniProt sequence and annotation, so they cover
  all of them. The files here are named for this count.
* **2,656 / 2,689** — the accessions with an AlphaFold model. The `structure`,
  `local_structure` and `fused_late` maps are built from 3-D structures, so they
  cover only these.

Each panel on the explorer page states its own protein count.

## Provenance

The membership was recovered from the archived explorer page's own payload on
2026-08-25; the original run directories no longer hold their inputs. The
selection came from an Arcadia run and is published here deliberately, so that
the cohort behind each map is inspectable rather than implicit.
