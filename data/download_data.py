# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

from pathlib import Path
from warnings import simplefilter

import numpy as np
import openml
import pandas as pd
from pandas import DataFrame, Series

DATA_DIR = ROOT / "data"


def find_correct_data_versions() -> DataFrame:
    """
    Notes
    -----
    We drop the APSFailure dataset (dataset_id = 8) because it is an extreme
    outlier with a massive amount of missing values and enormous class imbalance.
    The number of missing values means it fundamentally cannot be treated the same
    as the other datasets (e.g. impute missing values with median, split sanely),
    so there is not much point including it for comparison to others.
    """
    PAPER_TABLE = DATA_DIR / "paper_datasets.csv"
    DATA_TABLE = DATA_DIR / "datasets.csv"

    paper = pd.read_csv(PAPER_TABLE)
    names = paper["dataset_name"]
    datasets = openml.datasets.list_datasets(output_format="dataframe")
    datasets = datasets[datasets["name"].isin(names)]
    # print(datasets)

    # Check for missing
    paper_names = set(names.unique().tolist())
    online = set(datasets.name.unique().tolist())
    missing = paper_names.difference(online)
    if len(missing) > 0:
        print("Missing:", missing)

    renamer = {
        "NumberOfInstances": "n_sample",
        "NumberOfFeatures": "n_feat",
        "MajorityClassSize": "n_majority_cls",
        "MinorityClassSize": "n_minority_cls",
        "NumberOfClasses": "n_cls",
        "NumberOfInstancesWithMissingValues": "n_nan_rows",
        "did": "id",
    }
    datasets.rename(columns=renamer, inplace=True)
    select = ["id", "name", "version", *(list(renamer.values())[:-1])]
    df = datasets.loc[:, select]
    df["n_sample"] = df["n_sample"].astype(int)
    df["n_feat"] = df["n_feat"].astype(int)

    pd.options.display.max_rows = None  # type: ignore
    pd.options.display.max_info_rows = None  # type: ignore
    pd.options.display.expand_frame_repr = True  # type: ignore
    paper_counts = paper.loc[:, "n_samples"]
    paper_counts.index = paper["dataset_name"].str.lower()  # type: ignore
    paper_samples = df.name.apply(lambda name: paper_counts.loc[name.lower()])
    df.insert(3, "paper_n_sample", paper_samples)

    paper_feats = paper.loc[:, "n_features"]
    paper_feats.index = paper["dataset_name"].str.lower()  # type: ignore
    paper_n_feats = df.name.apply(lambda name: paper_feats.loc[name.lower()])
    df.insert(5, "paper_n_feat", paper_n_feats)
    print(df)

    matching = df.loc[df.paper_n_sample == df.n_sample]
    matching = matching.loc[df.paper_n_feat == df.n_feat]
    # print("\n\nMatching:")
    # print(matching)
    # print(matching.shape)

    latest_matching = (
        matching.sort_values(by=["name", "version"], ascending=True)
        .groupby("name")
        .apply(lambda grp: grp.nlargest(1, columns="version"))
        .droplevel(["name"])  # type: ignore
        .drop(columns="id")
    )
    latest_matching.index.name = "did"
    # print(latest_matching)
    # print(latest_matching.shape)
    latest_matching.to_csv(DATA_TABLE, index=True)
    latest_matching["did"] = latest_matching.index.values
    return latest_matching.loc[:, ["did", "version"]]


if __name__ == "__main__":
    simplefilter("ignore", FutureWarning)
    to_dl = find_correct_data_versions()
    for dataset_id, version in zip(to_dl["did"].to_list(), to_dl["version"].to_list()):
        ds = openml.datasets.get_dataset(dataset_id=dataset_id, version=version)
        X: DataFrame
        y: Series
        X, y, categoricals, attributes = ds.get_data(dataset_format="dataframe")
        df = X.copy()
        for column_idx in np.where(categoricals)[0]:
            df.iloc[:, column_idx] = df.iloc[:, column_idx].astype("category")
        column = ds.default_target_attribute
        if y is None:
            y = X[column]
            X.drop(columns=column, inplace=True)
        df["__target"] = y
        if column in df:
            df.drop(columns=column, inplace=True)

        # fname = f"{ds.dataset_id}_v{ds.version}_{ds.name}.json"
        # outdir = DATA_DIR / "json"
        fname = f"{ds.dataset_id}_v{ds.version}_{ds.name}.parquet"
        outdir = DATA_DIR / "parquet"
        outdir.mkdir(exist_ok=True)
        outfile = outdir / fname
        # df.to_json(outfile, indent=2)
        df.to_parquet(outfile)
        print(f"Saved data to {outfile}")
