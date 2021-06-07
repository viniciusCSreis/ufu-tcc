import os
import pandas as pd

if __name__ == '__main__':
    v2_result_path = '../imagens_cra/result/v2'
    v2_folders = os.listdir(v2_result_path)
    v2_folders = list(filter(lambda f: str(f).startswith("unet_"), v2_folders))

    summary = []

    for folder in v2_folders:
        summary_path = os.path.join(v2_result_path, folder, "metrics_train-size.csv")
        metrics_csv = pd.read_csv(summary_path)

        train = metrics_csv[:527]
        validation = metrics_csv[527:]

        summary.append(
            [
                folder,
                train["IoU"].mean(),
                train["IoU"].median(),
                train["IoU"].min(),
                train["IoU"].max(),
                train["IoU"].size,
                validation["IoU"].mean(),
                validation["IoU"].median(),
                validation["IoU"].min(),
                validation["IoU"].max(),
                validation["IoU"].size,
            ]
        )

    summary_df = pd.DataFrame(summary, columns=[
        "rede",
        "train_mean", "train_median", "train_min", "train_max", "train_size",
        "val_mean", "val_median", "val_min", "val_max", "val_size",
    ])

    summary_df.to_csv(os.path.join(v2_result_path, "summary_train-size.csv"))
    print("finish")


