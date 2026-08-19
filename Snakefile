import os
import sys
from pathlib import Path

from ProteinCartography import config_utils

# The package's modules import each other flat (`from spaces.base import ...`),
# which works when snakemake runs them as `python ProteinCartography/foo.py`
# because that puts the package directory on the path. Importing one of them
# *here*, in the snakemake process, needs the same thing done explicitly.
sys.path.insert(0, str(Path(workflow.basedir) / "ProteinCartography"))

import config_schema  # noqa: E402


# Default pipeline configuration parameters are in this file
# If you create a new yml file and use the --configfile flag,
# options in that new file overwrite the defaults.
configfile: "./config.yml"


# the mode in which to run the pipeline (either 'search' or 'cluster')
MODE = config_utils.Mode(config["mode"])

# basic configuration parameters common to both search and cluster modes
INPUT_DIR = Path(config["input_dir"])
OUTPUT_DIR = Path(config["output_dir"])
ANALYSIS_NAME = config["analysis_name"]
TAXON_FOCUS = config["taxon_focus"]
PLOTTING_MODES = config["plotting_modes"]

# in search mode, SEARCH_MODE_INPUT_PROTIDS are the IDs of the input proteins that are used
# for the similarity searches; in cluster mode, this is simply an empty list
# in both modes, KEY_PROTIDS are the IDs of the proteins that are highlighted
# in the final analysis plots
SEARCH_MODE_INPUT_PROTIDS, KEY_PROTIDS = config_utils._get_protids(config)

# the features-override file is an optional user-provided TSV file used to provide metadata
# for specific proteins; in search mode, it overrides any metadata downloaded by UniProt
FEATURES_OVERRIDE_FILE = config_utils._get_features_override_file(config)

# the features file is specific to, and required in, cluster mode; it provides uniprot-like metadata
# for all of the input proteins (since, in cluster mode, metadata is not downloaded from UniProt)
FEATURES_FILE = config_utils._get_features_file(config)

BENCHMARKS_DIR = OUTPUT_DIR / "benchmarks"

# results from running blastp with the input proteins
BLAST_RESULTS_DIR = OUTPUT_DIR / "blast_results"

# results from calling the foldseek web API with the input proteins
FOLDSEEK_RESULTS_DIR = OUTPUT_DIR / "foldseek_results"

# metadata related to the hits (proteins) from blast and foldseek
PROTEIN_FEATURES_DIR = OUTPUT_DIR / "protein_features"

# PDBs downloaded from AlphaFold
DOWNLOADED_PROTEIN_STRUCTURES_DIR = OUTPUT_DIR / "protein_structures"

# the directory containing the PDB files to assess and cluster (this is mode-dependent)
ANALYZED_PROTEIN_STRUCTURES_DIR = (
    DOWNLOADED_PROTEIN_STRUCTURES_DIR if MODE == config_utils.Mode.SEARCH else INPUT_DIR
)

# results from running foldseek to cluster the PDBs
FOLDSEEK_CLUSTERING_DIR = OUTPUT_DIR / "foldseek_clustering_results"

# results from calculating TM-scores with foldseek
FOLDSEEK_TMSCORES_DIR = OUTPUT_DIR / "key_protid_tmscores_results"

# final output results (plots and aggregated TSV files)
FINAL_RESULTS_DIR = OUTPUT_DIR / "final_results"

# multi-space outputs; empty unless the config defines `spaces`
BLOCKS_DIR = OUTPUT_DIR / "blocks"
SPACES_DIR = OUTPUT_DIR / "spaces"
COREGISTRATION_DIR = OUTPUT_DIR / "coregistration"
ENRICHMENT_DIR = OUTPUT_DIR / "enrichment"

# Foldseek's 3Di structural alphabet, one row per analyzed structure. Produced
# only when a block uses the `threedi` provider.
THREEDI_DESCRIPTORS_FILENAME = "3di_descriptors.tsv"

# The multi-space configuration. A config with no `blocks`/`spaces` keys is
# translated into the single structure space the pipeline has always built, and
# `MULTISPACE_ENABLED` stays false so none of the new rules enter the DAG. That
# is what keeps the default run byte-identical: the rules below are unreachable
# unless someone asks for them.
#
# Validation happens here rather than inside a rule so that an invalid config --
# in particular one that tries to fuse an overlay-only signal -- fails before
# any work is done, instead of four hours in. See docs/adr/0003-the-fusable-flag.md.
MULTISPACE_CONFIG = config_schema.from_legacy(config)
MULTISPACE_ENABLED = bool(config.get("spaces"))
MULTISPACE_TARGETS = []
if MULTISPACE_ENABLED:
    for space_id, space in sorted(MULTISPACE_CONFIG.spaces.items()):
        for reducer in space.reducers:
            # Built by concatenation, never an f-string: under PEP 701 `make
            # format` rewrites the doubled braces that make a literal wildcard.
            MULTISPACE_TARGETS.append(SPACES_DIR / space_id / ("embedding_" + reducer + ".tsv"))

# Diagnostics run for every space a config defines, rather than behind their own
# key. They are the caveats on a map, and a caveat nobody opted into is the only
# kind worth having -- an opt-in diagnostic is read by exactly the people who
# already suspected the problem. Empty when there are no spaces, so the default
# DAG is untouched.
DIAGNOSTICS_TARGETS = []
if MULTISPACE_ENABLED:
    for space_id in sorted(MULTISPACE_CONFIG.spaces):
        DIAGNOSTICS_TARGETS.append(SPACES_DIR / space_id / "diagnostics.json")

# One file, all data embedded, no server (ADR 0005). Unreachable without a
# `spaces:` key, like every other rule in this work, and it depends on the
# diagnostics rather than on the embeddings alone: what it refuses to draw is
# decided by the diagnostics, and building it from coordinates only would give
# a page that renders every space as if it were trustworthy.
EXPLORER_TARGETS = []
if MULTISPACE_ENABLED:
    EXPLORER_TARGETS.append(FINAL_RESULTS_DIR / (ANALYSIS_NAME + "_explorer.html"))

# Co-registration compares spaces pairwise. Empty unless `coregistration.compare`
# names at least two of them, so asking for spaces does not silently buy a
# comparison as well.
COREGISTERED_SPACES = list(MULTISPACE_CONFIG.coregistration.compare) if MULTISPACE_ENABLED else []
COREGISTRATION_ENABLED = len(COREGISTERED_SPACES) > 1
COREGISTRATION_PAIR_FILES = []
# Procrustes needs both layouts in the same coordinate system, so it needs one
# reducer that every compared space ran. Picking the first shared one keeps the
# choice deterministic; when there is none, the entry point records that the
# disparity was not computed rather than inventing a comparison.
COREGISTRATION_REDUCER = None
if COREGISTRATION_ENABLED:
    for space_a, space_b in (
        (a, b) for i, a in enumerate(COREGISTERED_SPACES) for b in COREGISTERED_SPACES[i + 1 :]
    ):
        COREGISTRATION_PAIR_FILES.append(
            COREGISTRATION_DIR / (space_a + "__vs__" + space_b + ".tsv")
        )
    shared_reducers = set.intersection(
        *(set(MULTISPACE_CONFIG.spaces[s].reducers) for s in COREGISTERED_SPACES)
    )
    if shared_reducers:
        COREGISTRATION_REDUCER = sorted(shared_reducers)[0]

# Cluster enrichment is gated on its own key rather than on `spaces`, because it
# needs neither: it takes a cluster table and an annotation table, and a legacy
# cluster-mode run produces both. A config that names no annotation column gets
# no rule, so the default DAG is unchanged.
ENRICHMENT_ENABLED = MULTISPACE_CONFIG.enrichment.enabled

# search-mode-specific parameters
# note: although these parameters are only used in search mode, we can assume they exist here
# because they are defined in the base config file, which snakemake always loads
# (and which the user-defined config can only override, not replace)
MAX_BLAST_HITS = int(config["max_blast_hits"])
BLAST_EVALUE = float(config["blast_evalue"])
BLAST_WORD_SIZE = int(config["blast_word_size"])
BLAST_WORD_SIZE_BACKOFF = int(config["blast_word_size_backoff"])
BLAST_NUM_ATTEMPTS = int(config["blast_num_attempts"])
BLAST_TIMEOUT_SECONDS = float(config["blast_timeout_seconds"])
BLAST_DATABASE = config["blast_database"]
FOLDSEEK_SERVER_URL = config["foldseek_server_url"]
FOLDSEEK_DATABASES = config["foldseek_databases"]
MAX_FOLDSEEK_HITS = int(config["max_foldseek_hits"])
# Read through the cohort config so that `cohort.max_structures` is honored when
# set, while a legacy config that only knows `max_structures` keeps working --
# `from_legacy` copies the old key across when the new one is absent.
MAX_STRUCTURES = MULTISPACE_CONFIG.cohort.max_structures
COHORT_SELECTION = MULTISPACE_CONFIG.cohort.selection
COHORT_MEASURE = MULTISPACE_CONFIG.cohort.measure
# Significance ranking needs a score per candidate, and building that table
# means reading every raw search result. The default rule needs no scores, so
# the rule that builds it stays out of the DAG entirely unless it is asked for.
COHORT_NEEDS_SIGNIFICANCE = COHORT_SELECTION == "significance"
MIN_LENGTH = int(config["min_length"])
MAX_LENGTH = int(config["max_length"])
UNIPROT_ADDITIONAL_FIELDS = config["uniprot_additional_fields"]

# Parallel domain map (gated). Protein-path directories and rules above are unchanged.
DOMAIN_MAP = config_utils._get_domain_map(config)
MIN_DOMAIN_LENGTH = config_utils._get_min_domain_length(config)
USER_DOMAINS_FILE = config_utils._get_user_domains_file(config)

# The domain path truncates its own hit list, so it makes a second cohort
# decision that `cohort.selection` has to reach or silently contradict.
# `domain_download_pdbs` passes `--selection` for that reason.
#
# `significance` is the one rule it cannot honor: ranking by significance needs
# the per-accession table `aggregate_hit_significance` builds, and there is no
# domain-side equivalent. Refusing the run would be wrong -- a search-mode
# config with `selection: significance` is valid, predates the domain path, and
# the protein map is unaffected. So give the domain path a *stated* reproducible
# rule rather than letting it fall through to the script's `as_filtered`
# default, and say so where somebody will see it. `accession` is the right
# substitute: it is in `cohort.REPRODUCIBLE_RULES` and needs no extra input.
#
# What this removes is the silence. Before, one config truncated the protein map
# by significance and the domain map by UniProt's response order, with no error
# and no cohort report on the domain side to notice it from.
# See docs/adr/0008-cohort-selection.md.
DOMAIN_COHORT_SELECTION = COHORT_SELECTION
if DOMAIN_MAP != "off" and COHORT_SELECTION == "significance":
    DOMAIN_COHORT_SELECTION = "accession"
    print(
        "[cohort] selection 'significance' cannot be applied to the domain map "
        "(there is no domain-side significance table), so the domain map uses "
        "'accession'. Both are reproducible; they are not the same cut. Set "
        "`domain_map: off` if the two maps must share one rule.",
        file=sys.stderr,
    )

DOMAIN_DIR = OUTPUT_DIR / "domain_path"
DOMAIN_BLAST_DIR = DOMAIN_DIR / "blast_results"
DOMAIN_FOLDSEEK_DIR = DOMAIN_DIR / "foldseek_results"
DOMAIN_HIT_STRUCTURES_DIR = DOMAIN_DIR / "hit_structures"
DOMAIN_STRUCTURES_DIR = DOMAIN_DIR / "domain_structures"
DOMAIN_CLUSTERING_DIR = OUTPUT_DIR / "foldseek_clustering_results_domain"
DOMAIN_TMSCORES_DIR = OUTPUT_DIR / "key_domain_tmscores_results"
DOMAIN_FEATURES_DIR = DOMAIN_DIR / "features"
DOMAIN_GATE_PROTIDS = (
    SEARCH_MODE_INPUT_PROTIDS
    if MODE == config_utils.Mode.SEARCH
    else [path.stem for path in sorted(INPUT_DIR.glob("*.pdb"))]
)


wildcard_constraints:
    plotting_mode="|".join(PLOTTING_MODES),
    protid="|".join(SEARCH_MODE_INPUT_PROTIDS + KEY_PROTIDS),
    # Constrained to the ids the config actually defines, and to no id at all
    # when multi-space is off. Without this a `{space_id}` wildcard would match
    # greedily across path separators, because snakemake wildcards are regexes
    # and `.+` happily eats a `/`.
    block_id="|".join(MULTISPACE_CONFIG.block_ids()) if MULTISPACE_ENABLED else "^$",
    space_id="|".join(MULTISPACE_CONFIG.space_ids()) if MULTISPACE_ENABLED else "^$",
    reducer="|".join(sorted(config_schema.LEGACY_PLOTTING_MODES)),


rule make_pdb:
    """
    Use the ESMFold API query to generate a PDB file from a fasta file.

    Note: we mark the input FASTA file as `ancient` to prevent snakemake from running this rule
    when the PDB file does exist but has a last-modified time prior to that of the FASTA file;
    because an extant PDB file is, by definition, user-provided, we can assume that it is correct.
    """
    input:
        fasta_file=ancient(INPUT_DIR / "{protid}.fasta"),
    output:
        pdb_file=INPUT_DIR / "{protid}.pdb",
    benchmark:
        BENCHMARKS_DIR / "{protid}.make_pdb.txt"
    conda:
        "envs/web_apis.yml"
    shell:
        """
        python ProteinCartography/esmfold_apiquery.py --input {input.fasta_file} --output {output.pdb_file}
        """


rule copy_pdb:
    """
    Copies existing or generated PDBs to the protein structures folder.
    """
    input:
        INPUT_DIR / "{protid}.pdb",
    output:
        DOWNLOADED_PROTEIN_STRUCTURES_DIR / "{protid}.pdb",
    shell:
        """
        cp {input} {output}
        """


rule run_blast:
    """
    Using files located in the input directory, run `blastp` using the remote BLAST API.

    Large proteins will cause remote BLAST to fail;
    you can still perform a manual BLAST search to get around this.
    """
    input:
        fasta_file=INPUT_DIR / "{protid}.fasta",
    output:
        blast_results=BLAST_RESULTS_DIR / "{protid}.blast_results.tsv",
    benchmark:
        BENCHMARKS_DIR / "{protid}.run_blast.txt"
    conda:
        "envs/blast.yml"
    shell:
        """
        python ProteinCartography/run_blast.py \
          --query {input.fasta_file} \
          --out {output.blast_results} \
          --max_target_seqs {MAX_BLAST_HITS} \
          --word_size {BLAST_WORD_SIZE} \
          --word_size_backoff {BLAST_WORD_SIZE_BACKOFF} \
          --num_attempts {BLAST_NUM_ATTEMPTS} \
          --evalue {BLAST_EVALUE} \
          --timeout_seconds {BLAST_TIMEOUT_SECONDS} \
          --database {BLAST_DATABASE}
        """


rule extract_blast_hits:
    """
    Extracts the top hits from the BLAST results file.
    """
    input:
        blast_results=BLAST_RESULTS_DIR / "{protid}.blast_results.tsv",
    output:
        blast_hits=BLAST_RESULTS_DIR / "{protid}.blast_hits.refseq.txt",
    benchmark:
        BENCHMARKS_DIR / "{protid}.extract_blast_hits.txt"
    conda:
        "envs/pandas.yml"
    shell:
        """
        python ProteinCartography/extract_blast_hits.py \
            --input {input.blast_results} \
            --output {output.blast_hits}
        """


rule map_refseq_ids:
    """
    Map a list of RefSeq IDs to UniProt IDs using the Uniprot ID mapping API or bioservices.
    """
    input:
        blast_hits=rules.extract_blast_hits.output.blast_hits,
    output:
        # The from/to pairs behind the hit list. That list alone loses which
        # RefSeq accession became which UniProt entry, and the correspondence is
        # the only way to key a BLAST e-value to a cohort candidate.
        #
        # Declared only when `aggregate_hit_significance` is in the DAG to read
        # it. An output nothing consumes is not free: see the comment on
        # `download_pdbs`, where the same shape silently drops a rule.
        **(
            {"refseq_mapping": BLAST_RESULTS_DIR / "{protid}.blast_hits.mapping.tsv"}
            if COHORT_NEEDS_SIGNIFICANCE
            else {}
        ),
        blast_hits_uniprot_ids=BLAST_RESULTS_DIR / "{protid}.blast_hits.uniprot.txt",
    params:
        # A function of `output`, not a formatted string: a params string is
        # substituted into the shell command verbatim and is never re-expanded,
        # so a `{protid}` written here would reach the command line literally.
        mapping_args=lambda wildcards, output: (
            "--mapping-output " + output.refseq_mapping if COHORT_NEEDS_SIGNIFICANCE else ""
        ),
    benchmark:
        BENCHMARKS_DIR / "{protid}.map_refseq_ids.txt"
    conda:
        "envs/web_apis.yml"
    shell:
        """
        python ProteinCartography/map_refseq_ids.py \
            --input {input.blast_hits} \
            --output {output.blast_hits_uniprot_ids} \
            {params.mapping_args}
        """


rule run_foldseek:
    """
    Queries Foldseek using the web API.
    The script accepts an input file ending in '.pdb' and returns an output file ending in '.tar.gz'.
    The script also accepts a `--mode` flag of either '3diaa' (default) or 'tmalign'.
    After running, untars the files and extracts hits.

    Note: the foldseek web API returns a limited number of hits; up to 1000 per database
    """
    input:
        pdb_file=INPUT_DIR / "{protid}.pdb",
    output:
        foldseek_output=FOLDSEEK_RESULTS_DIR / "{protid}.fsresults.tar.gz",
        m8_files_dir=directory(FOLDSEEK_RESULTS_DIR / "{protid}"),
        m8_files=expand(
            FOLDSEEK_RESULTS_DIR / "{{protid}}" / "alis_{db}.m8",
            db=FOLDSEEK_DATABASES,
        ),
    conda:
        "envs/web_apis.yml"
    benchmark:
        BENCHMARKS_DIR / "{protid}.run_foldseek.txt"
    shell:
        """
        python ProteinCartography/foldseek_apiquery.py \
            --input {input.pdb_file} \
            --output {output.foldseek_output} \
            --server {FOLDSEEK_SERVER_URL} \
            --database {FOLDSEEK_DATABASES}
        tar -xvf {output.foldseek_output} -C {output.m8_files_dir}
        """


rule extract_foldseek_hits:
    input:
        m8_files=rules.run_foldseek.output.m8_files,
    output:
        foldseek_hits=FOLDSEEK_RESULTS_DIR / "{protid}.foldseek_hits.txt",
    conda:
        "envs/pandas.yml"
    shell:
        """
        python ProteinCartography/extract_foldseek_hits.py \
            --input {input.m8_files} \
            --output {output.foldseek_hits} \
            --max-num-hits {MAX_FOLDSEEK_HITS}
        """


rule aggregate_foldseek_fraction_seq_identity:
    """
    Pulls the foldseek fraction sequence identity (fident) from the Foldseek results files
    for each input protein.

    This will probably be replaced in the future by an all-v-all sequence identity comparison
    using FAMSA, WITCH, or other approach.
    """
    input:
        m8_files=rules.run_foldseek.output.m8_files,
    output:
        fident_features=PROTEIN_FEATURES_DIR / "{protid}_fident_features.tsv",
    benchmark:
        BENCHMARKS_DIR / "{protid}.aggregate_foldseek_fident.txt"
    conda:
        "envs/pandas.yml"
    shell:
        """
        python ProteinCartography/aggregate_foldseek_fraction_seq_identity.py \
            --input {input.m8_files} \
            --output {output.fident_features} \
            --protid {wildcards.protid}
        """


rule aggregate_hit_significance:
    """
    Best e-value and bit score per hit, across every query that found it.

    Only in the DAG when `cohort.selection` is `significance` -- see the
    conditional input on `download_pdbs`. The default cohort rule needs no
    scores, so the default run does not build this and its output tree is
    unchanged.

    Note that this is an e-value and not a TM-score, which is what ADR 0008
    originally asked for. The Foldseek web API's .m8 output has no TM-score
    column, and the TM-scores the pipeline is built around come from the local
    all-versus-all run, which happens after the structures are downloaded. See
    the header of hit_significance.py.
    """
    input:
        # Guarded by the same flag that declares the output. This rule only
        # enters the DAG under `significance` selection, but its input block is
        # evaluated at parse time for every config, so an unguarded reference
        # to an output that is no longer declared is an AttributeError on the
        # default path.
        **(
            {
                "refseq_mapping": expand(
                    rules.map_refseq_ids.output.refseq_mapping,
                    protid=SEARCH_MODE_INPUT_PROTIDS,
                )
            }
            if COHORT_NEEDS_SIGNIFICANCE
            else {}
        ),
        m8_files=expand(rules.run_foldseek.output.m8_files, protid=SEARCH_MODE_INPUT_PROTIDS),
        blast_results=expand(rules.run_blast.output.blast_results, protid=SEARCH_MODE_INPUT_PROTIDS),
    output:
        significance=PROTEIN_FEATURES_DIR / "hit_significance.tsv",
    benchmark:
        BENCHMARKS_DIR / "aggregate_hit_significance.txt"
    conda:
        "envs/pandas.yml"
    shell:
        """
        python ProteinCartography/hit_significance.py \
            --foldseek-m8 {input.m8_files} \
            --blast-results {input.blast_results} \
            --refseq-mapping {input.refseq_mapping} \
            --output {output.significance}
        """


rule aggregate_hits:
    """
    Take all Uniprot ID lists and make them one big ID list, removing duplicates.
    """
    input:
        expand(rules.map_refseq_ids.output.blast_hits_uniprot_ids, protid=SEARCH_MODE_INPUT_PROTIDS),
        expand(rules.extract_foldseek_hits.output.foldseek_hits, protid=SEARCH_MODE_INPUT_PROTIDS),
    output:
        aggregated_hits=PROTEIN_FEATURES_DIR / "aggregated_hits.txt",
    benchmark:
        BENCHMARKS_DIR / "aggregate_hits.txt"
    conda:
        "envs/pandas.yml"
    shell:
        """
        python ProteinCartography/aggregate_hits.py --input {input} --output {output}
        """


rule fetch_uniprot_metadata:
    """
    Query Uniprot for the aggregated hits and download all metadata as a big ol' TSV.
    """
    input:
        rules.aggregate_hits.output.aggregated_hits,
    output:
        uniprot_features=PROTEIN_FEATURES_DIR / "uniprot_features.tsv",
    benchmark:
        BENCHMARKS_DIR / "fetch_uniprot_metadata.txt"
    conda:
        "envs/web_apis.yml"
    shell:
        """
        python ProteinCartography/fetch_uniprot_metadata.py \
            --input {input} \
            --output {output.uniprot_features} \
            --additional-fields {UNIPROT_ADDITIONAL_FIELDS}
        """


rule filter_aggregated_hits:
    """
    Use the metadata features from Uniprot to filter hits
    based on sequence status, fragment, and size.
    """
    input:
        rules.fetch_uniprot_metadata.output.uniprot_features,
    output:
        filtered_aggregated_hits=PROTEIN_FEATURES_DIR / "filtered_aggregated_hits.txt",
    benchmark:
        BENCHMARKS_DIR / "filter_aggregated_hits.txt"
    conda:
        "envs/pandas.yml"
    shell:
        """
        python ProteinCartography/filter_aggregated_hits.py \
            --input {input} \
            --output {output.filtered_aggregated_hits} \
            --min-length {MIN_LENGTH} \
            --max-length {MAX_LENGTH} \
            --excluded-protids {SEARCH_MODE_INPUT_PROTIDS}
        """


checkpoint download_pdbs:
    """
    Download all PDB files from AlphaFold.

    This rule decides the cohort: the hit list is truncated to `max_structures`
    here, and every space and diagnostic downstream is conditioned on that cut.
    The `cohort_report` output records what was discarded and whether the
    retained set is reproducible. See docs/adr/0008-cohort-selection.md.

    The two metadata inputs are read for the report only. Both are already
    ancestors of this rule through `filter_aggregated_hits`, so naming them adds
    an edge the DAG already had and does not add a job. Cluster mode does not
    run this rule at all -- the user supplies the structures, so there is no
    cohort decision to report.
    """
    input:
        **(
            {"significance": rules.aggregate_hit_significance.output.significance}
            if COHORT_NEEDS_SIGNIFICANCE
            else {}
        ),
        filtered_aggregated_hits=rules.filter_aggregated_hits.output.filtered_aggregated_hits,
        uniprot_features=rules.fetch_uniprot_metadata.output.uniprot_features,
        aggregated_hits=rules.aggregate_hits.output.aggregated_hits,
    params:
        cohort_report_args=lambda wildcards, output: (
            "--cohort-report " + output.cohort_report if MULTISPACE_ENABLED else ""
        ),
        # Reads the path off `input` rather than rebuilding it. The same file is
        # declared by `aggregate_hit_significance` a hundred lines above and was
        # spelled out a second time here, so the two could drift: snakemake would
        # still satisfy the input and hand the script a path that no longer
        # exists. Two spellings of one path is the defect family that produced
        # four silent explorer defects; `spaces/layout.py` closes the
        # cross-module instances and this closes the one inside this file.
        significance_args=lambda wildcards, input: (
            "--significance-table "
            + str(input.significance)
            + " --significance-measure "
            + COHORT_MEASURE
            if COHORT_NEEDS_SIGNIFICANCE
            else ""
        ),
    output:
        # Declared only when `diagnose_space` is in the DAG to read it, and the
        # reason is not tidiness. This rule is a *checkpoint*. An output that no
        # job requests is never a reason to re-run it, so on an output tree
        # produced before this branch -- structures present, report absent --
        # snakemake leaves the checkpoint alone, `checkpoints.download_pdbs.get`
        # raises, `get_pdb_filepaths` contributes no `copy_pdb` job, and the run
        # proceeds *silently* without the query proteins. Reproduced against
        # `36a38c7`: 17 jobs with `copy_pdb`, 16 without.
        **(
            {"cohort_report": PROTEIN_FEATURES_DIR / "cohort_report.json"}
            if MULTISPACE_ENABLED
            else {}
        ),
        protein_structures_dir=directory(DOWNLOADED_PROTEIN_STRUCTURES_DIR),
    benchmark:
        BENCHMARKS_DIR / "download_pdbs.txt"
    conda:
        "envs/web_apis.yml"
    shell:
        """
        python ProteinCartography/download_pdbs.py \
            --input {input.filtered_aggregated_hits} \
            --output {output.protein_structures_dir} \
            --max-structures {MAX_STRUCTURES} \
            --selection {COHORT_SELECTION} \
            --uniprot-features {input.uniprot_features} \
            --candidates-before-filtering {input.aggregated_hits} \
            {params.cohort_report_args} \
            {params.significance_args}
        """


def get_pdb_filepaths(wildcards):
    """
    Returns a list of all of the PDB files to use for the clustering analysis.

    In search mode, this function references the `download_pdbs` checkpoint, triggering it to run,
    and then returns a list of all of the resulting downloaded PDB files
    as well as the PDB files corresponding to the input proteins.

    In cluster mode, this function simply returns the list of all PDB files in the input directory.
    """
    if MODE == config_utils.Mode.SEARCH:
        # note: referencing the `download_pdbs` checkpoint here is essential,
        # because this is what 'tells' snakemake to run the checkpoint
        pdb_dirpath = checkpoints.download_pdbs.get(**wildcards).output.protein_structures_dir
        pdb_filepaths = sorted(Path(pdb_dirpath).glob("*.pdb"))

        # append the paths to the PDB files corresponding to the input proteins
        # note: this triggers the `copy_pdb` rule to copy the input PDB files from `INPUT_DIR`
        # to `DOWNLOADED_PROTEIN_STRUCTURES_DIR`
        pdb_filepaths += expand(
            DOWNLOADED_PROTEIN_STRUCTURES_DIR / "{protid}.pdb", protid=SEARCH_MODE_INPUT_PROTIDS
        )

    elif MODE == config_utils.Mode.CLUSTER:
        # in cluster mode, we do not need to download any PDB files
        # (as they are provided by the user), so we do not reference the `download_pdbs` checkpoint
        pdb_filepaths = sorted(INPUT_DIR.glob("*.pdb"))

    return pdb_filepaths


rule assess_pdbs:
    """
    Calculates the quality of all PDBs
    """
    input:
        get_pdb_filepaths,
    output:
        pdb_features=PROTEIN_FEATURES_DIR / "pdb_features.tsv",
    benchmark:
        BENCHMARKS_DIR / "assess_pdbs.txt"
    conda:
        "envs/plotting.yml"
    shell:
        """
        python ProteinCartography/assess_pdbs.py \
            --input {ANALYZED_PROTEIN_STRUCTURES_DIR} \
            --output {output.pdb_features}
        """


rule foldseek_clustering:
    """
    Runs foldseek all-v-all TM-score comparison and foldseek clustering on all of the PDB files.
    """
    input:
        get_pdb_filepaths,
    output:
        all_by_all_tmscores=FOLDSEEK_CLUSTERING_DIR / "all_by_all_tmscore_pivoted.tsv",
        struclusters_features=FOLDSEEK_CLUSTERING_DIR / "struclusters_features.tsv",
        foldseek_database=FOLDSEEK_CLUSTERING_DIR / "temp" / "temp_db",
    conda:
        "envs/foldseek.yml"
    resources:
        mem_mb=32 * 1000,
    threads: 16
    benchmark:
        BENCHMARKS_DIR / "foldseek_clustering.txt"
    shell:
        """
        python ProteinCartography/foldseek_clustering.py \
            --query-folder {ANALYZED_PROTEIN_STRUCTURES_DIR} \
            --results-folder {FOLDSEEK_CLUSTERING_DIR}
        """


rule copy_key_protid_pdbs:
    """
    Copies existing or generated PDBs corresponding to the 'key' protids
    into a separate folder for the Foldseek TM-score calculation.

    Note: the PDB files are copied within the shell script of this rule
    (rather than at the snakemake level via a parametrized rule)
    because snakemake does not allow an input file to be in a directory
    that is also itself an output of another rule.
    Here, the PDB files copied by this rule may be the output of the `copy_pdb` or `make_pdb` rules,
    but they are in the `DOWNLOADED_PROTEIN_STRUCTURES_DIR` directory,
    which is itself the output of the `download_pdbs` rule.
    """
    input:
        get_pdb_filepaths,
    output:
        dirpath=directory(FOLDSEEK_TMSCORES_DIR),
    shell:
        """
        mkdir -p "{output.dirpath}"
        for protid in {KEY_PROTIDS}; do
        cp "{ANALYZED_PROTEIN_STRUCTURES_DIR}/${{protid}}.pdb" "{output.dirpath}"
        done
        """


rule calculate_key_protid_tmscores:
    """
    Generates complete TM-score comparisons for each
    input protein against all proteins in the dataset.
    """
    input:
        pdb_dirpath=rules.copy_key_protid_pdbs.output.dirpath,
        foldseek_database=rules.foldseek_clustering.output.foldseek_database,
    output:
        key_protid_tmscores=PROTEIN_FEATURES_DIR / "key_protid_tmscore_features.tsv",
    conda:
        "envs/foldseek.yml"
    benchmark:
        BENCHMARKS_DIR / "calculate_key_protid_tmscores.txt"
    shell:
        """
        python ProteinCartography/calculate_key_protid_tmscores.py \
            --query-database {input.foldseek_database} \
            --target-folder {input.pdb_dirpath} \
            --results-folder {input.pdb_dirpath} \
            --features-file {output.key_protid_tmscores}
        """


rule dim_reduction:
    """
    Perform dimensionality reduction, saving as an embedding matrix and a TSV
    Write a set of functions to return Dataframes for interactive compute
    Write helper functions to save the dataframes only called by main()
    """
    input:
        rules.foldseek_clustering.output.all_by_all_tmscores,
    output:
        all_by_all_tmscores=FOLDSEEK_CLUSTERING_DIR
        / "all_by_all_tmscore_pivoted_{plotting_mode}.tsv",
    conda:
        "envs/analysis.yml"
    benchmark:
        BENCHMARKS_DIR / "{plotting_mode}.dim_reduction.txt"
    shell:
        """
        python ProteinCartography/dim_reduction.py --input {input} --mode {wildcards.plotting_mode}
        """


rule multispace_config:
    """
    Write the run's config as JSON for the block and space scripts to read.

    Not a convenience. Those scripts run in `envs/analysis.yml`, which has no
    PyYAML, so they could not read the config at all -- `compute_block` failed at
    import the first time it was executed. Adding PyYAML to that environment
    would change its hash and force a fresh solve of the one environment whose
    package versions decide the pipeline's numeric output, which this repository
    has already been bitten by once. Writing JSON avoids the parser instead.

    `run:` rather than `shell:`, so it executes in snakemake's own environment
    and needs no conda environment of its own.
    """
    output:
        resolved=OUTPUT_DIR / "multispace_config.json",
    run:
        import json

        os.makedirs(os.path.dirname(output.resolved), exist_ok=True)
        with open(output.resolved, "w") as handle:
            json.dump(dict(config), handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")


rule extract_3di_descriptors:
    """
    Extract foldseek's 3Di structural alphabet for every analyzed structure.

    Its own rule rather than part of `compute_block` because it needs the
    foldseek binary, which lives in a different conda environment from the one
    the block providers run in. Same arrangement as `foldseek_clustering`: the
    tool writes a file, the provider reads it.

    Only in the DAG when a block actually uses the `threedi` provider.
    """
    input:
        pdb_files=get_pdb_filepaths,
    output:
        descriptors=PROTEIN_FEATURES_DIR / THREEDI_DESCRIPTORS_FILENAME,
        # foldseek writes a `.dbtype` sidecar next to its output. Declared so
        # snakemake tracks and cleans it rather than leaving it as litter that
        # nothing in the workflow accounts for.
        dbtype=PROTEIN_FEATURES_DIR / (THREEDI_DESCRIPTORS_FILENAME + ".dbtype"),
    conda:
        "envs/foldseek.yml"
    benchmark:
        BENCHMARKS_DIR / "extract_3di_descriptors.txt"
    shell:
        """
        foldseek structureto3didescriptor {input.pdb_files} {output.descriptors}
        """


#: Providers that read the UniProt features table. Keyed on the provider rather
#: than the block id, for the reason recorded in
#: config_schema.NOT_FUSABLE_PROVIDERS: the block id is a name the user chooses,
#: so a table keyed on it would work only for users who happened to pick the
#: expected one.
FEATURES_TABLE_PROVIDERS = ("biophys", "domains")


def uniprot_features_table():
    """The features table for this mode.

    Search mode fetches it into the output directory; cluster mode takes the
    user's file from the input directory. Same split as
    `get_aggregate_features_input`, and the reason the path cannot come from the
    config: it is a property of the run, not of the block.
    """
    if MODE == config_utils.Mode.SEARCH:
        return rules.fetch_uniprot_metadata.output.uniprot_features
    return FEATURES_FILE


def get_block_extra_inputs(wildcards):
    """Inputs a specific block's provider needs beyond the similarity matrix."""
    block = MULTISPACE_CONFIG.blocks.get(wildcards.block_id)
    if block is None:
        return []
    if block.provider == "threedi":
        return [PROTEIN_FEATURES_DIR / THREEDI_DESCRIPTORS_FILENAME]
    if block.provider in FEATURES_TABLE_PROVIDERS:
        return [uniprot_features_table()]
    return []


def get_block_provider_inputs(wildcards):
    """The `--provider-input NAME=PATH` arguments for this block.

    Separate from the input list because snakemake's `{input}` expansion gives
    the provider a bare path with no indication of what it is. A provider that
    reads two files would have to index into that list positionally, which is
    the same defect as reading a labeled matrix by position (ADR 0007).
    """
    block = MULTISPACE_CONFIG.blocks.get(wildcards.block_id)
    if block is not None and block.provider in FEATURES_TABLE_PROVIDERS:
        return "--provider-input features_file=" + str(uniprot_features_table())
    return ""


rule compute_block:
    """
    Compute one representation and write it to the block store.

    Additive: nothing in the legacy pipeline depends on this, and the rule is
    unreachable unless the config defines `spaces`. A block whose provider is
    unavailable records a skip rather than failing the DAG, so a missing
    optional dependency costs you that block and nothing else.
    """
    input:
        all_by_all_tmscores=rules.foldseek_clustering.output.all_by_all_tmscores,
        resolved_config=rules.multispace_config.output.resolved,
        provider_inputs=get_block_extra_inputs,
    output:
        manifest=BLOCKS_DIR / "{block_id}" / "manifest.json",
        protids=BLOCKS_DIR / "{block_id}" / "protids.txt",
    params:
        provider_inputs=get_block_provider_inputs,
    conda:
        "envs/analysis.yml"
    benchmark:
        BENCHMARKS_DIR / "compute_block_{block_id}.txt"
    shell:
        """
        python ProteinCartography/compute_block.py \
            --configfile {input.resolved_config} \
            --block-id {wildcards.block_id} \
            --output-dir {OUTPUT_DIR} \
            {params.provider_inputs}
        """


def get_space_block_inputs(wildcards):
    """The block manifests a space needs before it can be reduced."""
    space = MULTISPACE_CONFIG.spaces[wildcards.space_id]
    return [str(BLOCKS_DIR / block_id / "manifest.json") for block_id in space.blocks]


rule reduce_space:
    """
    Reduce one space to coordinates, through the same reducer core that the
    legacy `dim_reduction` rule uses.

    Sharing that core is the point: two implementations of the pipeline's PCA
    would eventually disagree, and the disagreement would look like a
    scientific result.
    """
    input:
        blocks=get_space_block_inputs,
        resolved_config=rules.multispace_config.output.resolved,
    output:
        embedding=SPACES_DIR / "{space_id}" / "embedding_{reducer}.tsv",
    conda:
        "envs/analysis.yml"
    benchmark:
        BENCHMARKS_DIR / "reduce_space_{space_id}_{reducer}.txt"
    shell:
        """
        python ProteinCartography/reduce_space.py \
            --configfile {input.resolved_config} \
            --space-id {wildcards.space_id} \
            --reducer {wildcards.reducer} \
            --output-dir {OUTPUT_DIR}
        """


def get_cohort_report_input(wildcards):
    """The cohort report, in search mode only.

    Cluster mode makes no cohort decision -- the user supplies the structures --
    so there is no report to read and its absence is correct rather than a gap.
    """
    if MODE != config_utils.Mode.SEARCH:
        return []
    return [str(PROTEIN_FEATURES_DIR / "cohort_report.json")]


def get_cohort_report_arg(wildcards):
    paths = get_cohort_report_input(wildcards)
    return "--cohort-report " + paths[0] if paths else ""


def get_space_embeddings(wildcards):
    """Every layout this space produced, for the faithfulness diagnostic."""
    space = MULTISPACE_CONFIG.spaces[wildcards.space_id]
    return [
        str(SPACES_DIR / wildcards.space_id / ("embedding_" + reducer + ".tsv"))
        for reducer in sorted(space.reducers)
    ]


def get_space_embedding_args(wildcards):
    """`--embedding REDUCER=PATH` per reducer.

    Named rather than positional, for the reason `coregister --embedding` is:
    snakemake hands the shell a bare list of paths and working out which
    reducer produced which by position is the same defect as reading a labeled
    matrix by position (ADR 0007).
    """
    space = MULTISPACE_CONFIG.spaces[wildcards.space_id]
    return " ".join(
        "--embedding "
        + reducer
        + "="
        + str(SPACES_DIR / wildcards.space_id / ("embedding_" + reducer + ".tsv"))
        for reducer in sorted(space.reducers)
    )


def get_coregistration_block_inputs(wildcards):
    """Every block behind every compared space."""
    needed = []
    for space_id in COREGISTERED_SPACES:
        for block_id in MULTISPACE_CONFIG.spaces[space_id].blocks:
            needed.append(str(BLOCKS_DIR / block_id / "manifest.json"))
    return sorted(set(needed))


def get_coregistration_embeddings(wildcards):
    """The 2-D embeddings the Procrustes comparison needs, or none."""
    if COREGISTRATION_REDUCER is None:
        return []
    return [
        str(SPACES_DIR / space_id / ("embedding_" + COREGISTRATION_REDUCER + ".tsv"))
        for space_id in COREGISTERED_SPACES
    ]


def get_coregistration_embedding_args(wildcards):
    """`--embedding SPACE_ID=PATH` per compared space.

    Named rather than positional for the reason `--provider-input` is: snakemake
    hands the shell a bare list of paths, and working out which is which by
    position is the same defect as reading a labeled matrix by position
    (ADR 0007).
    """
    if COREGISTRATION_REDUCER is None:
        return ""
    return " ".join(
        "--embedding "
        + space_id
        + "="
        + str(SPACES_DIR / space_id / ("embedding_" + COREGISTRATION_REDUCER + ".tsv"))
        for space_id in COREGISTERED_SPACES
    )


rule coregister_spaces:
    """
    Compare every pair of co-registered spaces over one shared protein index.

    Additive and opt-in twice over: the rule is unreachable without `spaces`,
    and unreachable again unless `coregistration.compare` names two of them.
    """
    input:
        blocks=get_coregistration_block_inputs,
        embeddings=get_coregistration_embeddings,
        resolved_config=rules.multispace_config.output.resolved,
    output:
        summary=COREGISTRATION_DIR / "summary.tsv",
        index=COREGISTRATION_DIR / "index.json",
        pairs=COREGISTRATION_PAIR_FILES,
    params:
        embeddings=get_coregistration_embedding_args,
    conda:
        "envs/analysis.yml"
    benchmark:
        BENCHMARKS_DIR / "coregister_spaces.txt"
    shell:
        """
        python ProteinCartography/coregister.py \
            --configfile {input.resolved_config} \
            --output-dir {OUTPUT_DIR} \
            {params.embeddings}
        """


rule leiden_clustering:
    """
    Performs Leiden clustering on the data using scanpy's implementation.
    """
    input:
        rules.foldseek_clustering.output.all_by_all_tmscores,
    output:
        leiden_features=FOLDSEEK_CLUSTERING_DIR / "leiden_features.tsv",
    conda:
        "envs/analysis.yml"
    benchmark:
        BENCHMARKS_DIR / "leiden_clustering.txt"
    shell:
        """
        python ProteinCartography/leiden_clustering.py \
            --input {input} \
            --output {output.leiden_features}
        """


rule diagnose_space:
    """
    Say what this space's map can and cannot be read for.

    Censoring, cohort fairness, block redundancy and embedding faithfulness,
    written next to the map they qualify rather than into a log nobody keeps.

    The Leiden clustering is an input because cross-cluster edge retention is
    the censoring number worth reading, and it needs a partition to be about.
    It comes from the legacy path, which is the pipeline's only clustering
    today; when spaces cluster in their own right this should take that
    instead.
    """
    input:
        blocks=get_space_block_inputs,
        embeddings=get_space_embeddings,
        clusters=rules.leiden_clustering.output.leiden_features,
        cohort=get_cohort_report_input,
        resolved_config=rules.multispace_config.output.resolved,
    output:
        diagnostics=SPACES_DIR / "{space_id}" / "diagnostics.json",
    params:
        embeddings=get_space_embedding_args,
        cohort=get_cohort_report_arg,
    conda:
        "envs/analysis.yml"
    benchmark:
        BENCHMARKS_DIR / "diagnose_space_{space_id}.txt"
    shell:
        """
        python ProteinCartography/diagnose_space.py \
            --configfile {input.resolved_config} \
            --space-id {wildcards.space_id} \
            --output-dir {OUTPUT_DIR} \
            --clusters {input.clusters} \
            {params.embeddings} {params.cohort}
        """


rule calculate_concordance:
    """
    Currently, this subtracts the fraction sequence identity from the TM-score
    to get a measure of whether something is more similar in sequence or structure.

    We're working on developing some kind of test statistic that evaluates the significance
    of this difference from some expectation.
    """
    input:
        distance_features=rules.calculate_key_protid_tmscores.output.key_protid_tmscores,
        fident_features=rules.aggregate_foldseek_fraction_seq_identity.output.fident_features,
    output:
        concordance_features=PROTEIN_FEATURES_DIR / "{protid}_concordance_features.tsv",
    benchmark:
        BENCHMARKS_DIR / "{protid}.calculate_concordance.txt"
    conda:
        "envs/pandas.yml"
    shell:
        """
        python ProteinCartography/calculate_concordance.py \
            --tmscore-file {input.distance_features} \
            --fident-file {input.fident_features} \
            --output {output.concordance_features} \
            --protid {wildcards.protid}
        """


rule get_source_of_hits:
    """
    Checks the blast hits and foldseek hits to determine the source of each protein.
    """
    input:
        tm_scores=rules.foldseek_clustering.output.all_by_all_tmscores,
        hit_files=(
            expand(
                rules.extract_foldseek_hits.output.foldseek_hits,
                protid=SEARCH_MODE_INPUT_PROTIDS,
            )
            + expand(
                rules.map_refseq_ids.output.blast_hits_uniprot_ids,
                protid=SEARCH_MODE_INPUT_PROTIDS,
            )
        ),
    output:
        source_features=PROTEIN_FEATURES_DIR / "source_features.tsv",
    benchmark:
        BENCHMARKS_DIR / "get_source_of_hits.txt"
    conda:
        "envs/pandas.yml"
    shell:
        """
        python ProteinCartography/get_source_of_hits.py \
            --input {input.tm_scores} \
            --hit-files {input.hit_files} \
            --output {output.source_features} \
            --keyids {KEY_PROTIDS}
        """


def get_aggregate_features_input(*_):
    """
    Returns the paths to all TSV files that are used as input to the `aggregate_features` rule
    """
    # Inputs common to both search and cluster modes.
    common_inputs = [
        rules.assess_pdbs.output.pdb_features,
        rules.foldseek_clustering.output.struclusters_features,
        rules.leiden_clustering.output.leiden_features,
        rules.calculate_key_protid_tmscores.output.key_protid_tmscores,
    ]
    common_inputs += expand(
        rules.aggregate_foldseek_fraction_seq_identity.output.fident_features, protid=KEY_PROTIDS
    )
    common_inputs += expand(
        rules.calculate_concordance.output.concordance_features, protid=KEY_PROTIDS
    )

    search_mode_inputs = [
        rules.fetch_uniprot_metadata.output.uniprot_features,
        rules.get_source_of_hits.output.source_features,
    ]

    cluster_mode_inputs = [FEATURES_FILE]

    if MODE == config_utils.Mode.SEARCH:
        return common_inputs + search_mode_inputs
    elif MODE == config_utils.Mode.CLUSTER:
        return common_inputs + cluster_mode_inputs


rule aggregate_features:
    """
    Aggregate all TSV features provided by user in some specific directory, making one big TSV
    """
    input:
        get_aggregate_features_input,
    output:
        aggregated_features=FINAL_RESULTS_DIR / f"{ANALYSIS_NAME}_aggregated_features.tsv",
    benchmark:
        BENCHMARKS_DIR / "aggregate_features.txt"
    conda:
        "envs/pandas.yml"
    shell:
        """
        python ProteinCartography/aggregate_features.py \
            --input {input} \
            --output {output.aggregated_features} \
            --features-override-file {FEATURES_OVERRIDE_FILE}
        """


rule enrich_clusters:
    """
    Test what each cluster is made of, and write it as a table.

    Additive and opt-in: the rule is unreachable unless `enrichment` names at
    least one annotation column. It is gated on that key alone rather than on
    `spaces`, because a legacy cluster-mode run already produces both tables it
    needs.

    The clusters come from `leiden_features.tsv`, which is the pipeline's only
    clustering -- Leiden over the TM-score matrix -- so this describes the
    structure space rather than the multi-space map, however many spaces the
    run built. The clustering is named in every output row. See ADR 0012.

    `envs/analysis.yml` is reused rather than added to: the script needs only
    numpy and pandas, but that environment already exists in any run that gets
    here and adding a new one would be a fresh solve for nothing.
    """
    input:
        clusters=rules.leiden_clustering.output.leiden_features,
        annotations=rules.aggregate_features.output.aggregated_features,
        resolved_config=rules.multispace_config.output.resolved,
    output:
        table=ENRICHMENT_DIR / "cluster_enrichment.tsv",
        manifest=ENRICHMENT_DIR / "manifest.json",
    conda:
        "envs/analysis.yml"
    benchmark:
        BENCHMARKS_DIR / "enrich_clusters.txt"
    shell:
        """
        python ProteinCartography/enrich_clusters.py \
            --configfile {input.resolved_config} \
            --output-dir {OUTPUT_DIR} \
            --clusters {input.clusters} \
            --annotations {input.annotations}
        """


rule build_explorer:
    """
    One self-contained HTML file: every co-registered space, linked selection,
    and the diagnostics that say which panels may be read.

    Depends on the diagnostics, not just the embeddings. A page built from
    coordinates alone would draw an unreadable space exactly like a trustworthy
    one, which is the failure ADR 0005 item 5 and ADR 0014 both exist to
    prevent.

    `plot_interactive` keeps working and keeps emitting its existing filenames;
    this is additive.

    Placed after `aggregate_features` because it references
    `rules.aggregate_features`, and snakemake resolves `rules.` at parse time --
    the same ordering constraint that moved `diagnose_space` below
    `leiden_clustering` in group 8b.
    """
    input:
        diagnostics=DIAGNOSTICS_TARGETS,
        features=rules.aggregate_features.output.aggregated_features,
        resolved_config=rules.multispace_config.output.resolved,
    output:
        explorer=FINAL_RESULTS_DIR / (ANALYSIS_NAME + "_explorer.html"),
    conda:
        "envs/plotting.yml"
    benchmark:
        BENCHMARKS_DIR / "build_explorer.txt"
    shell:
        """
        python ProteinCartography/build_explorer.py \
            --configfile {input.resolved_config} \
            --output-dir {OUTPUT_DIR} \
            --analysis-name {ANALYSIS_NAME} \
            --output {output.explorer}
        """


rule plot_interactive:
    """
    Generate interactive scatter plot HTML programmatically based on user-input parameters
    Takes the TSV from rule aggregate_features and select default columns
    User should be able to call this module and pass their own functions
    to parse particular TSV columns
    Should have means to set a palette for each individual plot type, maybe as JSON?
    """
    input:
        tm_scores=rules.dim_reduction.output.all_by_all_tmscores,
        features=rules.aggregate_features.output.aggregated_features,
    output:
        html=FINAL_RESULTS_DIR / (ANALYSIS_NAME + "_aggregated_features_{plotting_mode}.html"),
    conda:
        "envs/plotting.yml"
    benchmark:
        BENCHMARKS_DIR / "{plotting_mode}.plot_interactive.txt"
    shell:
        """
        python ProteinCartography/plot_interactive.py \
            --dimensions {input.tm_scores} \
            --features {input.features} \
            --output {output.html} \
            --dimensions-type {wildcards.plotting_mode} \
            --keyids {KEY_PROTIDS} \
            --taxon-focus {TAXON_FOCUS}
        """


rule plot_similarity_leiden:
    """
    Plots a similarity score matrix for Leiden clusters.
    For each cluster, calculates the mean TM-score of all structures in that cluster
    versus all other clusters.
    The diagonal of the plot shows how similar proteins are within a given cluster.
    The other cells show how similar other clusters are to each other.
    """
    input:
        tm_scores=rules.foldseek_clustering.output.all_by_all_tmscores,
        features=rules.leiden_clustering.output.leiden_features,
    output:
        tsv=FINAL_RESULTS_DIR / f"{ANALYSIS_NAME}_leiden_similarity.tsv",
        html=FINAL_RESULTS_DIR / f"{ANALYSIS_NAME}_leiden_similarity.html",
    params:
        column="LeidenCluster",
    conda:
        "envs/plotting.yml"
    benchmark:
        BENCHMARKS_DIR / "plot_similarity_leiden.txt"
    shell:
        """
        python ProteinCartography/plot_cluster_similarity.py \
            --matrix-file {input.tm_scores} \
            --features-file {input.features} \
            --features-column {params.column} \
            --output-tsv {output.tsv} \
            --output-html {output.html}
        """


rule plot_similarity_strucluster:
    """
    Plots a similarity score matrix for Foldseek's structural clusters.
    For each cluster, calculates the mean TM-score of all structures in that cluster
    versus all other clusters.
    The diagonal of the plot shows how similar proteins are within a given cluster.
    The other cells show how similar other clusters are to each other.

    TODO (KC): this rule is almost identical to `plot_similarity_leiden`
    """
    input:
        tm_scores=rules.foldseek_clustering.output.all_by_all_tmscores,
        features=rules.foldseek_clustering.output.struclusters_features,
    output:
        tsv=FINAL_RESULTS_DIR / f"{ANALYSIS_NAME}_strucluster_similarity.tsv",
        html=FINAL_RESULTS_DIR / f"{ANALYSIS_NAME}_strucluster_similarity.html",
    params:
        column="StruCluster",
    conda:
        "envs/plotting.yml"
    benchmark:
        BENCHMARKS_DIR / "plot_similarity_strucluster.txt"
    shell:
        """
        python ProteinCartography/plot_cluster_similarity.py \
            --matrix-file {input.tm_scores} \
            --features-file {input.features} \
            --features-column {params.column} \
            --output-tsv {output.tsv} \
            --output-html {output.html}
        """


rule plot_semantic_analysis:
    """
    Plots a semantic analysis chart for groups within the data.
    """
    input:
        features=rules.aggregate_features.output.aggregated_features,
    output:
        pdf=FINAL_RESULTS_DIR / f"{ANALYSIS_NAME}_semantic_analysis.pdf",
        html=FINAL_RESULTS_DIR / f"{ANALYSIS_NAME}_semantic_analysis.html",
    params:
        agg_column="LeidenCluster",
        annot_column="'Protein names'",
    conda:
        "envs/plotting.yml"
    benchmark:
        BENCHMARKS_DIR / "plot_semantic_analysis.txt"
    shell:
        """
        python ProteinCartography/semantic_analysis.py \
            --features-file {input.features} \
            --agg-column {params.agg_column} \
            --annot-column {params.annot_column} \
            --output {output.pdf} \
            --interactive {output.html} \
            --analysis-name {ANALYSIS_NAME}
        """


rule plot_cluster_distributions:
    """
    Plots distributions of key values per cluster for each input protein.
    """
    input:
        features=rules.aggregate_features.output.aggregated_features,
    output:
        svg=FINAL_RESULTS_DIR / (ANALYSIS_NAME + "_{protid}_distribution_analysis.svg"),
    conda:
        "envs/plotting.yml"
    benchmark:
        BENCHMARKS_DIR / "plot_cluster_distributions_{protid}.txt"
    shell:
        """
        python ProteinCartography/plot_cluster_distributions.py \
            --input {input.features} \
            --output {output.svg} \
            --keyid {wildcards.protid}
        """


include: "Snakefile_domain"


rule all:
    """
    This is a pseudo-rule that defines the final outputs of the pipeline

    Note: this rule appears at the end, rather than the beginning, of the snakefile
    in order to allow the definition of its inputs in terms of the outputs of other rules
    (whose definitions must appear before this rule)
    """

    # we use `default_target` to tell snakemake that this is the first rule to run
    # (it otherwise defaults to running the first rule in the snakefile)
    default_target: True
    input:
        rules.plot_similarity_leiden.output.html,
        rules.plot_similarity_strucluster.output.html,
        rules.plot_semantic_analysis.output.html,
        rules.plot_semantic_analysis.output.pdf,
        expand(rules.plot_interactive.output.html, plotting_mode=PLOTTING_MODES),
        expand(rules.plot_cluster_distributions.output.svg, protid=KEY_PROTIDS),
        domain_final_outputs,
        # Empty unless the config defines `spaces`, so the default DAG is
        # exactly the one it has always been.
        MULTISPACE_TARGETS,
        # One per space, and empty for the same reason MULTISPACE_TARGETS is.
        DIAGNOSTICS_TARGETS,
        EXPLORER_TARGETS,
        # Empty again unless `coregistration.compare` names two spaces.
        [str(COREGISTRATION_DIR / "summary.tsv")] if COREGISTRATION_ENABLED else [],
        # And again unless `enrichment` names an annotation column.
        [str(ENRICHMENT_DIR / "cluster_enrichment.tsv")] if ENRICHMENT_ENABLED else [],
