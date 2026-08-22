#!/usr/bin/env python
import argparse
import os
import re

import constants
import pandas as pd
from hit_significance import TmalignOutputError, looks_like_tmalign

# only import these functions when using import *
__all__ = ["extract_foldseekhits"]

DEFAULT_EVALUE = 0.01


# parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", nargs="+", required=True, help="Takes .m8 file paths as input."
    )
    parser.add_argument("-o", "--output", required=True, help="Returns a .txt file as output.")
    parser.add_argument(
        "-e",
        "--evalue",
        type=float,
        default=DEFAULT_EVALUE,
        help="Sets maximum evalue for filtering.",
    )
    parser.add_argument(
        "-m",
        "--max-num-hits",
        type=int,
        help=(
            "Maximum number of hits to include in the output file. "
            "If not provided, all hits will be included."
        ),
    )

    args = parser.parse_args()

    return args


def extract_foldseekhits(
    input_files: list, output_file: str, evalue=DEFAULT_EVALUE, max_num_hits=None
):
    """
    Takes a list of input tabular Foldseek results files from the API query (ending in .m8).
    Generates a .txt file containing a list of unique accessions across all the .m8 files.

    Args:
        input_files (list): list of string paths to input files.
        output_file (str): path of destination file.
    """

    # empty df for collecting results
    dummy_df = pd.DataFrame()

    # iterate through results files, reading them
    for i, file in enumerate(input_files):
        # The size check has to come BEFORE the read, which is where it was not
        # (FOLLOWUPS #24). It is inert under the pinned pandas 2.0.1 -- reading
        # an empty file with `names=` supplied returns a 0-row frame rather than
        # raising -- so moving it changes no output today. It is moved anyway
        # because the guard only reads as a guard from here: a pandas that
        # raises on the empty read, or a caller that supplies no `names`, would
        # find the check sitting downstream of the thing it was meant to stop.
        if os.path.getsize(file) == 0:
            continue

        # load the file
        file_df = pd.read_csv(file, sep="\t", names=constants.FOLDSEEK_COLUMN_NAMES)

        # extract the model ID from the results target column
        file_df["modelid"] = file_df["target"].str.split(" ", expand=True)[0]

        # extract only models that contain AF model string
        # this will need to be changed in the future
        file_df = file_df[file_df["modelid"].str.contains("-F1-model")]

        # Refuse a tmalign-shaped file BEFORE filtering on the e-value column,
        # because in that mode the column is a TM-score and the filter's
        # polarity inverts: `evalue < 0.001` would keep the LEAST similar
        # structures and silently hand them to the map.
        #
        # One guard, one definition, two consumers: `hit_significance` has
        # refused this since it was written, and this module -- which is the one
        # that actually decides the cohort -- did not. The asymmetry was the
        # defect, not the missing check.
        if looks_like_tmalign(
            pd.to_numeric(file_df["evalue"], errors="coerce"),
            pd.to_numeric(file_df["bits"], errors="coerce"),
        ):
            evalues = pd.to_numeric(file_df["evalue"], errors="coerce").dropna()
            raise TmalignOutputError(
                f"{file} looks like `foldseek_apiquery.py --mode tmalign` output: its "
                f"e-value column runs {evalues.min():.4g} to {evalues.max():.4g}, entirely "
                "inside [0, 1], with bit scores at TM-score scale. In that mode the server "
                "reuses the same column positions for different quantities -- the 'e-value' "
                "is a TM-score -- so filtering it ascending keeps the LEAST similar "
                "structures. Refusing rather than guessing: the mode is not recorded in the "
                "output, so this file cannot be interpreted with confidence. The pipeline "
                "runs 3diaa mode, which this does support."
            )

        # filter by evalue
        file_df = file_df[file_df["evalue"] < evalue]

        # get the uniprot ID out from that target
        file_df["uniprotid"] = file_df["modelid"].apply(
            lambda x: re.findall("AF-(.*)-F1-model", x)[0]
        )

        # if it's the first results file, fill the dummy_df
        if i == 0:
            dummy_df = file_df
        # otherwise, add to the df
        else:
            dummy_df = pd.concat([dummy_df, file_df], axis=0)

    # extract unique uniprot IDs
    if dummy_df.empty:
        print(f"WARNING: No matching foldseek hits found in {input_files}.")
        hits = []
    else:
        hits = dummy_df["uniprotid"].unique()

    # if max_num_hits is provided, truncate the list
    if max_num_hits is not None:
        hits = hits[:max_num_hits]

    # save to a .txt file
    with open(output_file, "w+") as f:
        f.writelines(hit + "\n" for hit in hits)


# run this if called from the interpreter
def main():
    # parse arguments
    args = parse_args()

    # collect arguments individually
    input_files = args.input
    output_file = args.output
    evalue = args.evalue
    max_num_hits = args.max_num_hits

    if os.environ.get("PROTEINCARTOGRAPHY_SHOULD_USE_MOCKS") == "true":
        from tests.mock_domain_hits import maybe_write_per_domain_hits

        if maybe_write_per_domain_hits(output_file):
            return

    extract_foldseekhits(input_files, output_file, evalue=evalue, max_num_hits=max_num_hits)


# check if called from interpreter
if __name__ == "__main__":
    main()
