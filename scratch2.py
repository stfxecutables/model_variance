import numpy as np
import pandas as pd
from pandas import DataFrame


def acc1(lr: float, depth: int, is_adam: bool) -> float:
    max_depth = 101
    effective_depth = min(1, (depth / max_depth))
    opt_lr = 3e-4
    effective_lr = -np.abs(np.log10(lr) - np.log10(opt_lr)) / np.log10(opt_lr)
    max_acc = 0.95
    reduction = 0.98 if is_adam else 1.0
    return effective_depth * reduction * effective_lr * max_acc + 0.2


def acc2(lr: float, depth: int, is_adam: bool) -> float:
    max_depth = 101
    effective_depth = min(1, (depth / max_depth))
    opt_lr = 3e-4
    effective_lr = -np.abs(np.log10(lr) - np.log10(opt_lr)) / np.log10(opt_lr)
    max_acc = 0.95
    reduction = 0.98 if is_adam else 1.0
    return min(effective_depth * reduction * effective_lr * max_acc * 1.2, 1.0)


def local_sensitivity(df: DataFrame) -> float:
    ks = []
    c_max = df["acc"].max()
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            dlr = abs(
                np.log10(df.loc[i, "lr"])
                - np.log10(df.loc[j, "lr"])
                - np.log10(LRS.min()) / (np.log10(LRS.max()) - np.log(LRS.min()))
            )
            d_depth = abs(df.loc[i, "depth"] - df.loc[j, "depth"] - DEPTHS.min()) / (
                DEPTHS.max() - DEPTHS.min()
            )
            d_opt = np.sum(df.loc[i, "opt"] != df.loc[j, "opt"]) / len(OPTS)
            d_criterion = np.abs(df.loc[i, "acc"] - df.loc[j, "acc"])
            ks.append((d_criterion / c_max) / (dlr + d_depth + d_opt))
    return ks


LRS = np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
DEPTHS = np.array([34, 50, 101])
OPTS = ["AdamW", "SGD"]

if __name__ == "__main__":
    dfs1 = []
    dfs2 = []
    for lr in LRS:
        for depth in DEPTHS:
            for opt in OPTS:
                dfs1.append(
                    DataFrame(
                        {
                            "lr": lr,
                            "depth": depth,
                            "opt": opt,
                            "acc": acc1(lr, depth, opt == "AdamW"),
                        },
                        index=[0],
                    )
                )
                dfs2.append(
                    DataFrame(
                        {
                            "lr": lr,
                            "depth": depth,
                            "opt": opt,
                            "acc": acc2(lr, depth, opt == "AdamW"),
                        },
                        index=[0],
                    )
                )
    df1 = pd.concat(dfs1, axis=0, ignore_index=True)
    df2 = pd.concat(dfs2, axis=0, ignore_index=True)
    print(
        df1.sort_values(by="acc", ascending=True).to_markdown(
            tablefmt="simple", floatfmt=[".0e", "0.0f", "0.0f", "0.4f"], index=False
        )
    )
    print(np.max(local_sensitivity(df1)))
    print(np.max(local_sensitivity(df2)))
