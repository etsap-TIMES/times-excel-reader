import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pandas.core.frame import DataFrame
import pandas as pd
import pickle
from pathlib import Path
import os
import sys
import time
from typing import Dict, List
from . import datatypes
from . import excel
from . import transforms


def convert_xl_to_times(
    input_files: List[str],
    output_dir: str,
    config: datatypes.Config,
    use_pkl: bool,
    stop_after_read: bool = False,
) -> Dict[str, DataFrame]:
    pickle_file = "raw_tables.pkl"
    if use_pkl and os.path.isfile(pickle_file):
        raw_tables = pickle.load(open(pickle_file, "rb"))
        print(f"WARNING: Using pickled data not xlsx")
    else:
        raw_tables = []

        use_pool = True
        if use_pool:
            with ProcessPoolExecutor() as executor:
                for result in executor.map(excel.extract_tables, input_files):
                    raw_tables.extend(result)
        else:
            for f in input_files:
                result = excel.extract_tables(f)
                raw_tables.extend(result)
        pickle.dump(raw_tables, open(pickle_file, "wb"))
    print(
        f"Extracted {len(raw_tables)} tables,"
        f" {sum(table.dataframe.shape[0] for table in raw_tables)} rows"
    )

    if stop_after_read:
        # Convert absolute paths to relative paths to enable comparing raw_tables.txt across machines
        raw_tables.sort(key=lambda x: (x.filename, x.sheetname, x.range))
        input_dir = os.path.commonpath([t.filename for t in raw_tables])
        raw_tables = [strip_filename_prefix(t, input_dir) for t in raw_tables]

    dump_tables(raw_tables, os.path.join(output_dir, "raw_tables.txt"))
    if stop_after_read:
        return {}

    transform_list = [
        transforms.generate_dummy_processes,
        transforms.normalize_tags_columns_attrs,
        transforms.remove_fill_tables,
        lambda config, tables: [transforms.remove_comment_rows(t) for t in tables],
        lambda config, tables: [transforms.remove_comment_cols(t) for t in tables],
        transforms.remove_tables_with_formulas,  # slow
        transforms.process_transform_insert,
        transforms.process_processes,
        transforms.process_topology,
        transforms.process_flexible_import_tables,  # slow
        transforms.process_user_constraint_tables,
        transforms.process_commodity_emissions,
        transforms.process_commodities,
        transforms.process_transform_availability,
        transforms.process_time_slices,
        transforms.fill_in_missing_values,
        transforms.expand_rows_parallel,  # slow
        transforms.remove_invalid_values,
        transforms.process_time_periods,
        transforms.process_units,
        transforms.generate_all_regions,
        transforms.capitalise_attributes,
        transforms.apply_fixups,
        transforms.extract_commodity_groups,
        transforms.fill_in_missing_pcgs,
        transforms.generate_top_ire,
        transforms.include_tables_source,
        transforms.merge_tables,
        transforms.apply_more_fixups,
        transforms.process_years,
        transforms.process_uc_wildcards,
        transforms.process_wildcards,
        transforms.convert_aliases,
        transforms.rename_cgs,
        transforms.convert_to_string,
        lambda config, tables: dump_tables(
            tables, os.path.join(output_dir, "merged_tables.txt")
        ),
        lambda config, tables: produce_times_tables(config, tables),
    ]

    input = raw_tables
    output = {}
    for transform in transform_list:
        start_time = time.time()
        output = transform(config, input)
        end_time = time.time()
        print(
            f"transform {transform.__code__.co_name} took {end_time-start_time:.2f} seconds"
        )
        input = output

    print(
        f"Conversion complete, {len(output)} tables produced,"
        f" {sum(df.shape[0] for tablename, df in output.items())} rows"
    )

    return output


def write_csv_tables(tables: Dict[str, DataFrame], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    for item in os.listdir(output_dir):
        if item.endswith(".csv"):
            os.remove(os.path.join(output_dir, item))
    for tablename, df in tables.items():
        df.to_csv(os.path.join(output_dir, tablename + "_output.csv"), index=False)


def read_csv_tables(input_dir: str) -> Dict[str, DataFrame]:
    result = {}
    for filename in os.listdir(input_dir):
        result[filename.split(".")[0]] = pd.read_csv(
            os.path.join(input_dir, filename), dtype=str
        )
    return result


def compare(
    data: Dict[str, DataFrame], ground_truth: Dict[str, DataFrame], output_dir: str
):
    print(
        f"Ground truth contains {len(ground_truth)} tables,"
        f" {sum(df.shape[0] for _, df in ground_truth.items())} rows"
    )

    missing = set(ground_truth.keys()) - set(data.keys())
    missing_str = ", ".join(
        [f"{x} ({ground_truth[x].shape[0]})" for x in sorted(missing)]
    )
    if len(missing) > 0:
        print(f"WARNING: Missing {len(missing)} tables: {missing_str}")

    additional_tables = set(data.keys()) - set(ground_truth.keys())
    additional_str = ", ".join(
        [f"{x} ({data[x].shape[0]})" for x in sorted(additional_tables)]
    )
    if len(additional_tables) > 0:
        print(f"WARNING: {len(additional_tables)} additional tables: {additional_str}")
    # Additional rows starts as the sum of lengths of additional tables produced
    total_additional_rows = sum(len(data[x]) for x in additional_tables)

    total_gt_rows = 0
    total_correct_rows = 0
    for table_name, gt_table in sorted(
        ground_truth.items(), reverse=True, key=lambda t: len(t[1])
    ):
        total_gt_rows += len(gt_table)
        if table_name in data:
            data_table = data[table_name]

            # Remove .integer suffix added to duplicate column names by CSV reader (mangle_dupe_cols=False not supported)
            transformed_gt_cols = [col.split(".")[0] for col in gt_table.columns]
            data_cols = list(data_table.columns)

            if transformed_gt_cols != data_cols:
                print(
                    f"WARNING: Table {table_name} header incorrect, was"
                    f" {data_cols}, should be {transformed_gt_cols}"
                )
            else:
                # both are in string form so can be compared without any issues
                gt_rows = set(tuple(row) for row in gt_table.to_numpy().tolist())
                data_rows = set(tuple(row) for row in data_table.to_numpy().tolist())
                total_correct_rows += len(gt_rows.intersection(data_rows))
                additional = data_rows - gt_rows
                total_additional_rows += len(additional)
                missing = gt_rows - data_rows
                if len(additional) != 0 or len(missing) != 0:
                    print(
                        f"WARNING: Table {table_name} ({data_table.shape[0]} rows,"
                        f" {gt_table.shape[0]} GT rows) contains {len(additional)}"
                        f" additional rows and is missing {len(missing)} rows"
                    )
                if len(additional) != 0:
                    DataFrame(additional).to_csv(
                        os.path.join(output_dir, table_name + "_additional.csv"),
                        index=False,
                    )
                if len(missing) != 0:
                    DataFrame(missing).to_csv(
                        os.path.join(output_dir, table_name + "_missing.csv"),
                        index=False,
                    )

    print(
        f"{total_correct_rows / total_gt_rows :.1%} of ground truth rows present"
        f" in output ({total_correct_rows}/{total_gt_rows})"
        f", {total_additional_rows} additional rows"
    )


def produce_times_tables(
    config: datatypes.Config, input: Dict[str, DataFrame]
) -> Dict[str, DataFrame]:
    print(
        f"produce_times_tables: {len(input)} tables incoming,"
        f" {sum(len(value) for (_, value) in input.items())} rows"
    )
    result = {}
    used_tables = set()
    for mapping in config.times_xl_maps:
        if not mapping.xl_name in input:
            print(
                f"WARNING: Cannot produce table {mapping.times_name} because input table"
                f" {mapping.xl_name} does not exist"
            )
        else:
            used_tables.add(mapping.xl_name)
            df = input[mapping.xl_name].copy()
            # Filter rows according to filter_rows mapping:
            for filter_col, filter_val in mapping.filter_rows.items():
                if filter_col not in df.columns:
                    print(
                        f"WARNING: Cannot produce table {mapping.times_name} because input"
                        f" table {mapping.xl_name} does not contain column {filter_col}"
                    )
                    # TODO break this loop and continue outer loop?
                filter = set(x.lower() for x in {filter_val})
                i = df[filter_col].str.lower().isin(filter)
                df = df.loc[i, :]
            # TODO find the correct tech group
            if "techgroup" in mapping.xl_cols:
                df["techgroup"] = df["techname"]
            if not all(c in df.columns for c in mapping.xl_cols):
                missing = set(mapping.xl_cols) - set(df.columns)
                print(
                    f"WARNING: Cannot produce table {mapping.times_name} because input"
                    f" table {mapping.xl_name} does not contain the required columns"
                    f" - {', '.join(missing)}"
                )
            else:
                # Excel columns can be duplicated into multiple Times columns
                for times_col, xl_col in mapping.col_map.items():
                    df[times_col] = df[xl_col]
                cols_to_drop = [x for x in df.columns if not x in mapping.times_cols]
                df.drop(columns=cols_to_drop, inplace=True)
                df.drop_duplicates(inplace=True)
                df.reset_index(drop=True, inplace=True)
                # TODO this is a hack. Use pd.StringDtype() so that notna() is sufficient
                i = (
                    df[mapping.times_cols[-1]].notna()
                    & (df != "None").all(axis=1)
                    & (df != "").all(axis=1)
                )
                df = df.loc[i, mapping.times_cols]
                # Drop tables that are empty after filtering and dropping Nones:
                if len(df) == 0:
                    continue
                result[mapping.times_name] = df

    unused_tables = set(input.keys()) - used_tables
    if len(unused_tables) > 0:
        print(
            f"WARNING: {len(unused_tables)} unused tables: {', '.join(sorted(unused_tables))}"
        )

    return result


def write_dd_files(
    tables: Dict[str, DataFrame], config: datatypes.Config, output_dir: str
):
    os.makedirs(output_dir, exist_ok=True)
    for item in os.listdir(output_dir):
        if item.endswith(".dd"):
            os.remove(os.path.join(output_dir, item))

    def convert_set(df: DataFrame):
        has_description = "TEXT" in df.columns
        for row in df.itertuples(index=False):
            row_str = "'.'".join(
                (str(x) for k, x in row._asdict().items() if k != "TEXT")
            )
            desc = f" '{row.TEXT}'" if has_description else ""
            yield f"'{row_str}'{desc}\n"

    def convert_parameter(tablename: str, df: DataFrame):
        if "VALUE" not in df.columns:
            raise KeyError(f"Unable to find VALUE column in parameter {tablename}")
        for row in df.itertuples(index=False):
            val = row.VALUE
            row_str = "'.'".join(
                (str(x) for k, x in row._asdict().items() if k != "VALUE")
            )
            yield f"'{row_str}' {val}\n" if row_str else f"{val}\n"

    sets = {m.times_name for m in config.times_xl_maps if "VALUE" not in m.col_map}

    # Compute map fname -> tables: right now ALL_TS -> ts.dd, rest -> output.dd
    tables_in_file = {
        "ts.dd": ["ALL_TS"],
        "output.dd": [t for t in config.dd_table_order if t != "ALL_TS"],
    }

    for fname, tablenames in tables_in_file.items():
        with open(os.path.join(output_dir, fname), "w") as fout:
            for tablename in [t for t in tablenames if t in tables]:
                df = tables[tablename]
                if tablename in sets:
                    fout.write(f"SET {tablename}\n/\n")
                    lines = convert_set(df)
                else:
                    fout.write(f"PARAMETER\n{tablename} ' '/\n")
                    lines = convert_parameter(tablename, df)
                fout.writelines(sorted(lines))
                fout.write("\n/;\n")
    pass


def strip_filename_prefix(table, prefix):
    if isinstance(table, datatypes.EmbeddedXlTable):
        if table.filename.startswith(prefix):
            table.filename = table.filename[len(prefix) + 1 :]
    return table


def dump_tables(tables: List, filename: str) -> List:
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as text_file:
        for t in tables if isinstance(tables, List) else tables.items():
            if isinstance(t, datatypes.EmbeddedXlTable):
                tag = t.tag
                text_file.write(f"sheetname: {t.sheetname}\n")
                text_file.write(f"range: {t.range}\n")
                text_file.write(f"filename: {t.filename}\n")
                if t.uc_sets:
                    text_file.write(f"uc_sets: {t.uc_sets}\n")
                df = t.dataframe
            else:
                tag = t[0]
                df = t[1]
            text_file.write(f"tag: {tag}\n")
            types = ", ".join([f"{i} ({v})" for i, v in df.dtypes.items()])
            text_file.write(f"types: {types}\n")
            text_file.write(df.to_csv(index=False, lineterminator="\n"))
            text_file.write("\n" * 2)

    return tables


def main():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument(
        "input",
        nargs="*",
        help="Either an input directory, or a list of input xlsx files to process",
    )
    args_parser.add_argument(
        "--output_dir", type=str, default="output", help="Output directory"
    )
    args_parser.add_argument(
        "--ground_truth_dir",
        type=str,
        help="Ground truth directory to compare with output",
    )
    args_parser.add_argument("--dd", action="store_true", help="Output DD files")
    args_parser.add_argument(
        "--only_read",
        action="store_true",
        help="Read xlsx files and stop after outputting raw_tables.txt",
    )
    args_parser.add_argument("--use_pkl", action="store_true")
    args = args_parser.parse_args()

    config = datatypes.Config("times_mapping.txt", "times-info.json")

    if not isinstance(args.input, list) or len(args.input) < 1:
        print(f"ERROR: expected at least 1 input. Got {args.input}")
        sys.exit(1)
    elif len(args.input) == 1:
        assert os.path.isdir(args.input[0])
        input_files = [
            str(path)
            for path in Path(args.input[0]).rglob("*.xlsx")
            if not path.name.startswith("~")
        ]
        print(f"Loading {len(input_files)} files from {args.input[0]}")
    else:
        input_files = args.input

    if args.only_read:
        tables = convert_xl_to_times(
            input_files, args.output_dir, config, args.use_pkl, stop_after_read=True
        )
        sys.exit(0)

    tables = convert_xl_to_times(input_files, args.output_dir, config, args.use_pkl)

    if args.dd:
        write_dd_files(tables, config, args.output_dir)
    else:
        write_csv_tables(tables, args.output_dir)

    if args.ground_truth_dir:
        ground_truth = read_csv_tables(args.ground_truth_dir)
        compare(tables, ground_truth, args.output_dir)


if __name__ == "__main__":
    main()
