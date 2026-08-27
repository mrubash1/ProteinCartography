"""Tests for the multi-space configuration schema.

Two groups matter more than the rest:

* `from_legacy` -- an existing config.yml must keep working, unchanged. This is
  the mechanical half of the byte-identical-output promise.
* the `fusable` rejection -- the entire enforcement mechanism for ADR 0003. The
  tests check the *reason* reaches the user, not just that an error is raised,
  because the reason is the point.
"""

import inspect
import pathlib

import pytest
import pytest as _pytest
import yaml
from config_schema import (
    NOT_FUSABLE_REASONS,
    BlockConfig,
    CohortConfig,
    ConfigError,
    MultispaceConfig,
    SpaceConfig,
    from_legacy,
)
from config_utils import (
    RUNNABLE_MODES,
    SET_MODES,
    ProteinCartographyInputError,
    _get_foldseek_mode,
)
from reduce_space import params_for
from spaces.base import NotFusableError


def minimal(**overrides):
    data = {
        "blocks": {"tmscore": {"provider": "tmscore"}},
        "spaces": {"structure": {"blocks": ["tmscore"], "reducers": ["pca_umap"]}},
    }
    data.update(overrides)
    return data


# --------------------------------------------------------------------------
# legacy bridge
# --------------------------------------------------------------------------


def test_legacy_config_produces_one_block_and_one_space():
    config = from_legacy({"plotting_modes": ["pca_umap", "pca_tsne"]})
    assert config.block_ids() == ["tmscore"]
    assert config.space_ids() == ["structure"]
    assert config.spaces["structure"].reducers == ("pca_umap", "pca_tsne")
    assert config.from_legacy_config


def test_legacy_default_representation_is_profile():
    """`profile` is current behavior and is invariant to the column permutation."""
    config = from_legacy({"plotting_modes": ["pca_umap"]})
    assert config.blocks["tmscore"].representation == "profile"


def test_legacy_config_with_no_plotting_modes_still_works():
    config = from_legacy({})
    assert config.spaces["structure"].reducers == ("pca_umap",)


def test_the_real_shipped_config_yml_loads(tmp_path):
    """The actual default config in the repo must pass the new validator."""
    repo = pathlib.Path(__file__).resolve().parents[2]
    with open(repo / "config.yml") as fh:
        legacy = yaml.safe_load(fh)
    config = from_legacy(legacy)
    assert config.space_ids() == ["structure"]
    assert config.spaces["structure"].reducers == ("pca_tsne", "pca_umap")
    assert config.cohort.max_structures == 5000
    assert config.cohort.selection == "as_filtered"


def test_legacy_max_structures_reaches_the_cohort():
    config = from_legacy({"plotting_modes": ["pca_umap"], "max_structures": 10})
    assert config.cohort.max_structures == 10


def test_a_cohort_block_in_a_legacy_config_is_not_discarded():
    """Regression: `selection:` used to parse, validate, and do nothing.

    The legacy branch of `from_legacy` rebuilt the cohort mapping from scratch
    and copied only `max_structures` across, so a `cohort:` block in a config
    with no `blocks`/`spaces` keys -- which is every existing config -- was
    dropped on the floor. The DAG then silently used the default rule.
    """
    config = from_legacy({"plotting_modes": ["pca_umap"], "cohort": {"selection": "accession"}})
    assert config.cohort.selection == "accession"


def test_a_cohort_block_survives_alongside_blocks_and_spaces():
    """The same must hold on the modern branch, which took a different path."""
    config = from_legacy(minimal(cohort={"selection": "accession"}))
    assert config.cohort.selection == "accession"


def test_the_nested_max_structures_wins_over_the_top_level_one():
    """So a config can adopt `cohort:` without first deleting the old key."""
    config = from_legacy(
        {
            "plotting_modes": ["pca_umap"],
            "max_structures": 10,
            "cohort": {"max_structures": 99},
        }
    )
    assert config.cohort.max_structures == 99


def test_a_typo_in_the_cohort_block_is_rejected_rather_than_ignored():
    with pytest.raises(ConfigError, match="cohort"):
        from_legacy({"plotting_modes": ["pca_umap"], "cohort": {"selecton": "accession"}})


def test_legacy_rejects_an_unknown_plotting_mode():
    with pytest.raises(ConfigError, match="plotting_modes"):
        from_legacy({"plotting_modes": ["pca_magic"]})


def test_a_modern_config_passes_through_from_legacy():
    """Migration can be incremental: blocks/spaces win if present."""
    config = from_legacy(minimal())
    assert not config.from_legacy_config
    assert config.block_ids() == ["tmscore"]


def test_cohort_default_matches_the_shipped_default():
    assert MultispaceConfig.from_dict(minimal()).cohort.max_structures == 5000


# --------------------------------------------------------------------------
# ADR 0003 -- the fusable flag
# --------------------------------------------------------------------------


def test_known_overlay_only_signals_default_to_not_fusable():
    """A user who writes `taxonomy: {provider: ...}` gets the protection anyway."""
    config = MultispaceConfig.from_dict(
        {
            "blocks": {
                "tmscore": {"provider": "tmscore"},
                "taxonomy": {"provider": "uniprot_lineage"},
            },
            "spaces": {"structure": {"blocks": ["tmscore"]}},
        }
    )
    assert not config.blocks["taxonomy"].fusable
    assert "circular" in config.blocks["taxonomy"].not_fusable_reason


def test_fusing_a_non_fusable_block_is_rejected_with_the_reason():
    with pytest.raises(NotFusableError) as excinfo:
        MultispaceConfig.from_dict(
            {
                "blocks": {
                    "tmscore": {"provider": "tmscore"},
                    "taxonomy": {"provider": "uniprot_lineage"},
                },
                "spaces": {
                    "multiview": {
                        "blocks": ["tmscore", "taxonomy"],
                        "strategy": "late",
                    }
                },
            }
        )
    message = str(excinfo.value)
    assert "cannot be fused into space 'multiview'" in message
    assert "circular" in message
    assert "overlay" in message
    assert "adr/0003" in message


def test_struclusters_is_overlay_only_after_the_claim_b_finding():
    """Phase 0.5 claim B: sequence-weighted, so not structural ground truth."""
    assert "struclusters" in NOT_FUSABLE_REASONS
    reason = NOT_FUSABLE_REASONS["struclusters"]
    assert "amino-acid identity" in reason
    assert "circular" in reason


@pytest.mark.parametrize("block_id", sorted(NOT_FUSABLE_REASONS))
def test_every_known_overlay_only_signal_is_blocked_from_fusion(block_id):
    with pytest.raises(NotFusableError):
        MultispaceConfig.from_dict(
            {
                "blocks": {
                    "tmscore": {"provider": "tmscore"},
                    block_id: {"provider": "x"},
                },
                "spaces": {"fused": {"blocks": ["tmscore", block_id], "strategy": "late"}},
            }
        )


def test_a_single_block_space_may_use_an_overlay_only_signal():
    """Looking at a taxonomy-only layout is fine; letting it move a structure map is not."""
    config = MultispaceConfig.from_dict(
        {
            "blocks": {"taxonomy": {"provider": "uniprot_lineage"}},
            "spaces": {"tax_only": {"blocks": ["taxonomy"], "strategy": "none"}},
        }
    )
    assert config.space_ids() == ["tax_only"]


def test_fusable_false_without_a_known_reason_is_rejected():
    with pytest.raises(ConfigError) as excinfo:
        BlockConfig.from_dict("mystery", {"provider": "x", "fusable": False})
    assert "not_fusable_reason" in str(excinfo.value)


def test_explicit_reason_is_accepted_for_an_unknown_block():
    block = BlockConfig.from_dict(
        "mystery", {"provider": "x", "fusable": False, "not_fusable_reason": "because"}
    )
    assert not block.fusable
    assert block.not_fusable_reason == "because"


def test_overlay_only_blocks_are_enumerable():
    config = MultispaceConfig.from_dict(
        {
            "blocks": {
                "tmscore": {"provider": "tmscore"},
                "taxonomy": {"provider": "uniprot_lineage"},
            },
            "spaces": {"structure": {"blocks": ["tmscore"]}},
        }
    )
    assert list(config.overlay_only_blocks()) == ["taxonomy"]
    assert config.fusable_blocks() == ["tmscore"]


# --------------------------------------------------------------------------
# ADR 0007 -- representation: direct is gated
# --------------------------------------------------------------------------


def test_direct_representation_requires_verified_alignment():
    with pytest.raises(ConfigError) as excinfo:
        MultispaceConfig.from_dict(
            {
                "blocks": {"tmscore": {"provider": "tmscore", "representation": "direct"}},
                "spaces": {"structure": {"blocks": ["tmscore"]}},
            }
        )
    message = str(excinfo.value)
    assert "99.92% wrong" in message
    assert "representation: profile" in message
    assert "alignment_verified" in message


def test_direct_representation_allowed_once_verified():
    config = MultispaceConfig.from_dict(
        {
            "blocks": {
                "tmscore": {
                    "provider": "tmscore",
                    "representation": "direct",
                    "params": {"alignment_verified": True},
                }
            },
            "spaces": {"structure": {"blocks": ["tmscore"]}},
        }
    )
    assert config.blocks["tmscore"].representation == "direct"


def test_profile_representation_needs_no_gate():
    config = MultispaceConfig.from_dict(
        {
            "blocks": {"tmscore": {"provider": "tmscore", "representation": "profile"}},
            "spaces": {"structure": {"blocks": ["tmscore"]}},
        }
    )
    assert config.blocks["tmscore"].representation == "profile"


# --------------------------------------------------------------------------
# reference integrity and type checking
# --------------------------------------------------------------------------


def test_space_referencing_an_undefined_block_is_rejected():
    with pytest.raises(ConfigError) as excinfo:
        MultispaceConfig.from_dict(
            {
                "blocks": {"tmscore": {"provider": "tmscore"}},
                "spaces": {"s": {"blocks": ["plm"]}},
            }
        )
    message = str(excinfo.value)
    assert "'plm' is not defined" in message
    assert "tmscore" in message  # tells you what *is* defined


def test_coregistration_reference_must_exist():
    with pytest.raises(ConfigError, match="reference_space"):
        MultispaceConfig.from_dict(
            minimal(coregistration={"reference_space": "nope", "compare": []})
        )


def test_coregistration_compare_must_exist():
    with pytest.raises(ConfigError, match=r"compare\[0\]"):
        MultispaceConfig.from_dict(minimal(coregistration={"compare": ["nope"]}))


def test_unknown_top_level_space_key_is_rejected():
    """A misspelled key that is silently ignored is a setting you think you changed."""
    with pytest.raises(ConfigError) as excinfo:
        MultispaceConfig.from_dict(
            {
                "blocks": {"tmscore": {"provider": "tmscore"}},
                "spaces": {"s": {"blocks": ["tmscore"], "reducer": ["pca_umap"]}},
            }
        )
    assert "unknown key(s) ['reducer']" in str(excinfo.value)


def test_string_where_an_int_belongs_is_rejected_not_coerced():
    with pytest.raises(ConfigError, match="expected an integer"):
        MultispaceConfig.from_dict(minimal(cohort={"max_structures": "5000"}))


def test_boolean_where_a_number_belongs_is_rejected():
    with pytest.raises(ConfigError, match="expected a number"):
        MultispaceConfig.from_dict(
            {
                "blocks": {
                    "a": {"provider": "x"},
                    "b": {"provider": "y"},
                },
                "spaces": {"s": {"blocks": ["a", "b"], "strategy": "late", "weights": {"a": True}}},
            }
        )


def test_negative_max_structures_rejected():
    with pytest.raises(ConfigError, match="must be positive"):
        MultispaceConfig.from_dict(minimal(cohort={"max_structures": -1}))


def test_default_selection_is_the_honest_name_for_current_behavior():
    """ADR 0008: today's order is UniProt's response order, not accession order."""
    assert CohortConfig().selection == "as_filtered"


def test_accession_selection_is_available_but_opt_in():
    config = MultispaceConfig.from_dict(minimal(cohort={"selection": "accession"}))
    assert config.cohort.selection == "accession"


def test_unknown_selection_rule_rejected():
    with pytest.raises(ConfigError, match="selection"):
        MultispaceConfig.from_dict(minimal(cohort={"selection": "vibes"}))


def test_significance_selection_accepted():
    config = MultispaceConfig.from_dict(
        minimal(
            cohort={
                "selection": "significance",
                "significance_rule": {"blast": "best_evalue_across_queries"},
            }
        )
    )
    assert config.cohort.selection == "significance"


def test_the_default_significance_measure_is_the_one_search_mode_can_produce():
    """TM-score is not obtainable before the structures are downloaded (ADR 0008)."""
    config = MultispaceConfig.from_dict(minimal(cohort={"selection": "significance"}))
    assert config.cohort.measure == "evalue"


def test_a_named_significance_measure_is_used():
    config = MultispaceConfig.from_dict(
        minimal(cohort={"selection": "significance", "significance_rule": {"measure": "bits"}})
    )
    assert config.cohort.measure == "bits"


def test_an_unknown_significance_measure_is_rejected_at_parse_time():
    """Where it is *used* is after the searches have run, which is too late."""
    with pytest.raises(ConfigError, match="significance_rule.measure"):
        MultispaceConfig.from_dict(minimal(cohort={"significance_rule": {"measure": "pvalue"}}))


def test_subsample_fraction_bounds():
    with pytest.raises(ConfigError, match=r"must be in \(0, 1\]"):
        MultispaceConfig.from_dict(minimal(diagnostics={"subsample_fraction": 1.5}))


def test_provider_specific_keys_fold_into_params():
    """The provider is the authority on its own parameters, so extras pass through."""
    block = BlockConfig.from_dict("plm", {"provider": "plm", "model": "esm2_650m"})
    assert block.params["model"] == "esm2_650m"


def test_strategy_none_with_two_blocks_is_rejected():
    with pytest.raises(ConfigError) as excinfo:
        MultispaceConfig.from_dict(
            {
                "blocks": {"a": {"provider": "x"}, "b": {"provider": "y"}},
                "spaces": {"s": {"blocks": ["a", "b"], "strategy": "none"}},
            }
        )
    assert "co-registered spaces" in str(excinfo.value)


def test_a_full_multispace_config_validates():
    """The worked example from the plan, end to end."""
    config = MultispaceConfig.from_dict(
        {
            "cohort": {"max_structures": 5000, "selection": "significance"},
            "blocks": {
                "tmscore": {"provider": "tmscore", "representation": "profile"},
                "plm": {"provider": "plm", "model": "esm2_650m", "pooling": "mean"},
                "biophys": {
                    "provider": "biophys",
                    "normalization": "zscore_within",
                    "descriptors": ["gravy", "net_charge"],
                },
                "taxonomy": {"provider": "uniprot_lineage"},
            },
            "spaces": {
                "structure": {"blocks": ["tmscore"], "reducers": ["pca_umap", "pca_tsne"]},
                "sequence": {"blocks": ["plm"], "reducers": ["pca_umap"]},
                "struct_plus_seq": {
                    "blocks": ["tmscore", "plm"],
                    "strategy": "late",
                    "weights": {"tmscore": 1.0, "plm": 0.85},
                    "reducers": ["pca_umap"],
                },
                "multiview": {
                    "blocks": ["tmscore", "plm", "biophys"],
                    "strategy": "graph",
                    # `k`, not the paper's uppercase `K`. This example carried
                    # `K` from the day the schema was written and nothing ever
                    # read it, which is only visible now that `graph` exists and
                    # the params are checked against what it consumes.
                    "params": {"k": 20, "mu": 0.5},
                    "reducers": ["pca_umap"],
                },
            },
            "coregistration": {
                "reference_space": "structure",
                "compare": ["sequence", "struct_plus_seq", "multiview"],
                "k": 10,
            },
            "diagnostics": {"bootstrap_replicates": 20, "subsample_fraction": 0.8},
        }
    )
    assert config.space_ids() == [
        "multiview",
        "sequence",
        "struct_plus_seq",
        "structure",
    ]
    assert config.fusable_blocks() == ["biophys", "plm", "tmscore"]
    assert list(config.overlay_only_blocks()) == ["taxonomy"]
    assert config.spaces["struct_plus_seq"].weight_for("plm") == 0.85
    assert config.spaces["multiview"].params == {"k": 20, "mu": 0.5}


# ==========================================================================
# reducer_params. UMAP's `n_neighbors` was hardcoded at 80 and unreachable
# from a config; these prove the option exists, is read, and is guarded.
# ==========================================================================


def test_a_space_without_reducer_params_keeps_the_reducer_defaults():
    """The whole point of the field is that not using it changes nothing.

    80 is right at the production cohorts' scale and only questionable at a few
    hundred proteins, so the default must survive the option being added --
    otherwise every existing map moves and the parity suite is the thing that
    finds out.
    """
    space = SpaceConfig(id="structure", blocks=("tmscore",))
    assert space.reducer_params == {}


def test_reducer_params_rejects_a_parameter_the_reducer_will_not_read():
    """`n_neighbours` spelled the British way must not reach a manifest.

    This is FOLLOWUPS #29 and #32's failure -- a value recorded as configured
    while the run used the default -- and the cheap place to catch it is parse
    time. Same guard as `params`/STRATEGY_PARAMS, deliberately.
    """
    with pytest.raises(ConfigError) as excinfo:
        SpaceConfig(
            id="structure",
            blocks=("tmscore",),
            reducer_params={"pca_umap": {"n_neighbours": 15}},
        )
    message = str(excinfo.value)
    assert "n_neighbours" in message, "the rejection must name the offending key"
    assert "n_neighbors" in message, "and must list what the reducer does read"


def test_reducer_params_rejects_a_reducer_this_space_does_not_run():
    """Configuring `tsne` on a space that only runs `pca_umap` is a typo.

    It would otherwise sit in the config looking effective forever.
    """
    with pytest.raises(ConfigError):
        SpaceConfig(
            id="structure",
            blocks=("tmscore",),
            reducers=("pca_umap",),
            reducer_params={"tsne": {"perplexity": 5}},
        )


def test_params_for_reports_what_each_pipeline_reads():
    """The allow-list is derived from the pipeline's steps, not restated.

    A new pipeline built from existing steps inherits the right parameters
    without anyone remembering to update a second table.
    """
    assert "n_neighbors" in params_for("pca_umap")
    assert "n_neighbors" in params_for("umap")
    assert "perplexity" in params_for("pca_tsne")
    assert params_for("pca") == frozenset(), "PCA reads none of these"


# ==========================================================================
# `foldseek_mode`. The flag existed on `foldseek_apiquery.py` from the start and
# the Snakefile never passed it, so the mode was reachable by hand and
# unreachable by config, and the choice was recorded nowhere (PC-014 phase 3).
# ==========================================================================


def test_the_foldseek_mode_defaults_to_the_mode_the_pipeline_runs():
    assert _get_foldseek_mode({}) == "3diaa"
    assert _get_foldseek_mode({"foldseek_mode": "  3diaa  "}) == "3diaa"


def test_an_unknown_foldseek_mode_fails_at_config_parse_time():
    """Before the search, not after it -- the argument `SIGNIFICANCE_MEASURES`
    makes. `foldseek_apiquery.py` does check the value, but inside the rule,
    once per query protein, after the DAG has been built."""
    with _pytest.raises(ProteinCartographyInputError, match="foldseek_mode must be one of"):
        _get_foldseek_mode({"foldseek_mode": "3diAA"})
    with _pytest.raises(ProteinCartographyInputError, match="foldseek_mode must be one of"):
        _get_foldseek_mode({"foldseek_mode": "tm-align"})


def test_tmalign_is_refused_at_parse_time_and_not_left_to_die_in_the_rule():
    """It used to be ACCEPTED here. The value is a real Foldseek mode, so it
    passed a membership test against the API's vocabulary -- and then
    `extract_foldseek_hits.py:86` raised `TmalignOutputError` inside the rule,
    after the DAG was built and, in a real run, after the search. A validator
    whose whole argument is "fail before a four-hour search, not after it" was
    admitting the one value guaranteed to fail after it."""
    with _pytest.raises(ProteinCartographyInputError, match="foldseek_mode must be one of"):
        _get_foldseek_mode({"foldseek_mode": "tmalign"})


def test_the_refusal_explains_tmalign_rather_than_calling_it_a_typo():
    """A documented mode refused with the same message as a misspelling tells a
    user nothing. The two cases are distinguishable and are distinguished."""
    with _pytest.raises(ProteinCartographyInputError) as documented:
        _get_foldseek_mode({"foldseek_mode": "tmalign"})
    with _pytest.raises(ProteinCartographyInputError) as typo:
        _get_foldseek_mode({"foldseek_mode": "tm-align"})
    assert "TM-score" in str(documented.value)
    assert "TM-score" not in str(typo.value)


def test_the_two_vocabularies_are_both_real_and_do_not_collapse():
    """`SET_MODES` is what the API accepts; `RUNNABLE_MODES` is what this
    pipeline can consume. They are deliberately different, and a change that
    silently merged them would restore the accepts-then-dies behaviour."""
    assert SET_MODES == ["3diaa", "tmalign"]
    assert RUNNABLE_MODES == ["3diaa"]
    assert set(RUNNABLE_MODES) < set(SET_MODES)


def test_the_validator_reaches_no_module_outside_the_standard_library():
    """The predecessor of this test asserted that the string
    `"from foldseek_apiquery import SET_MODES"` appeared in the function's
    source -- pinning a MECHANISM while nothing tested its EFFECT. The mechanism
    was a deferred import, justified in a docstring as keeping the Snakefile
    free of a parse-time dependency on the HTTP stack, and it did not work: the
    Snakefile calls this function at parse time. This asserts the effect
    instead, and
    `test_snakefile_parses_without_optional_dependencies.py` asserts it end to
    end, in the subprocess where the real failure happened."""
    import ast
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(_get_foldseek_mode)))
    imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert imports == [], (
        "_get_foldseek_mode imports something. It is called while the Snakefile is "
        "being parsed, so anything it reaches must be in the standard library. "
        "Asserted on the AST rather than on the source text: the first version of "
        "this check looked for the substring 'import' and went red on the word "
        "'imports' inside the docstring."
    )


def test_the_shipped_config_states_the_mode():
    """`config.yml` is always loaded, so every existing user config inherits the
    key without editing anything."""
    root = pathlib.Path(__file__).resolve().parents[2]
    assert 'foldseek_mode: "3diaa"' in (root / "config.yml").read_text()


# ==========================================================================
# PC-036 phase 3 / WS5 -- the known-key sets, and the typo the top level ate
# ==========================================================================


def test_every_known_key_set_is_derived_from_its_dataclass():
    """Six sets used to be written out by hand beside the fields they mirrored.

    The drift runs in one direction only and is nasty when it happens: add a
    field, forget the set, and the loader REJECTS the key you just added with a
    message listing every key except the one the user typed. Measured before the
    change -- five of the six were already exactly the field names -- so this
    collapsed six copies rather than fixing six bugs, and this test is what stops
    a seventh copy appearing.
    """
    import dataclasses

    import config_schema as cs

    cases = [
        (cs.CohortConfig, set(), set()),
        (cs.BlockConfig, {"id"}, {"fusable_override_reason"}),
        (cs.SpaceConfig, {"id"}, set()),
        (cs.CoregistrationConfig, set(), set()),
        (cs.EnrichmentConfig, set(), set()),
        (cs.DiagnosticsConfig, set(), set()),
    ]
    for cls, drop, extra in cases:
        expected = {f.name for f in dataclasses.fields(cls)} - drop | extra
        assert cs._known_keys(cls, drop=tuple(drop), extra=tuple(extra)) == expected


def test_a_new_field_becomes_an_accepted_key_without_a_second_edit():
    """The property the derivation buys, stated as a behaviour rather than as a
    shape. `vocabulary_file` was added to BlockConfig this session and needed a
    second edit to the hand-written set; it would not now."""
    import config_schema as cs

    for name in ("vocabulary_file", "representation", "metric"):
        assert name in cs._known_keys(cs.BlockConfig, drop=("id",))


@pytest.mark.parametrize(
    "typo",
    ["coregistraton", "coregistrations", "diagnostic", "diagnostcs", "enrichmnt", "spaces_"],
)
def test_a_misspelled_top_level_key_is_refused_by_name(typo):
    """THE DEFECT. Every nested block rejects unknown keys; the top level cannot,
    because a legacy config.yml is full of keys this file knows nothing about.

    So `coregistraton:` was accepted in silence, the co-registration never ran,
    and nothing said why -- which is exactly what `_reject_unknown_keys`' own
    message warns about.
    """
    base = {
        "blocks": {"t": {"provider": "tmscore"}},
        "spaces": {"s": {"blocks": ["t"], "strategy": "none", "reducers": ["pca"]}},
    }
    with pytest.raises(ConfigError) as excinfo:
        from_legacy({**base, typo: {}})
    message = str(excinfo.value)
    assert typo in message, "the message must quote the key the user actually typed"
    assert "character(s) away from" in message


def test_an_unrelated_legacy_key_is_still_accepted():
    """The guard has to be narrow or it refuses every real config. `mode`,
    `input_dir` and `max_blast_hits` are nowhere near a multispace key."""
    base = {
        "blocks": {"t": {"provider": "tmscore"}},
        "spaces": {"s": {"blocks": ["t"], "strategy": "none", "reducers": ["pca"]}},
    }
    config = {
        **base,
        "mode": "search",
        "input_dir": "in",
        "max_blast_hits": 10,
        "plotting_modes": ["pca_umap"],
        "analysis_name": "x",
        "features_file": "f.tsv",
    }
    assert from_legacy(config) is not None


def test_no_shipped_config_trips_the_near_miss_guard():
    """The guard's threshold is a judgement, so it is checked against the configs
    that actually exist rather than argued about."""
    import glob

    import yaml

    root = pathlib.Path(__file__).resolve().parents[2]
    paths = sorted(
        glob.glob(str(root / "config.yml")) + glob.glob(str(root / "demo/*/config*.yml"))
    )
    assert paths, "no configs found to check the guard against"
    for path in paths:
        from_legacy(yaml.safe_load(open(path)))
