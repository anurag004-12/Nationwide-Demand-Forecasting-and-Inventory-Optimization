import argparse
import pandas as pd
import lightning.pytorch as pl
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss

from src.config import PROCESSED, OUTPUTS, MODELS, HORIZON
from src.metrics import score


def main(epochs):
    pl.seed_everything(42)                       # same result every run
    OUTPUTS.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)

    df = pd.read_parquet(PROCESSED / "features.parquet")

    # Use only recent history so training is faster
    df = df[df["t"] >= 1000].copy()
    df["time_idx"] = df["t"] - df["t"].min()     # TFT needs 0, 1, 2, ... integer time
    for col in ["id", "store_id", "item_id"]:
        df[col] = df[col].astype(str)
    df["dow_s"] = df["dow"].astype(str)
    df["month_s"] = df["month"].astype(str)
    df["sales"] = df["sales"].astype(float)

    cutoff = df["time_idx"].max() - HORIZON      # last 28 days = test period

    training = TimeSeriesDataSet(
        df[df["time_idx"] <= cutoff],
        time_idx="time_idx",
        target="sales",
        group_ids=["id"],
        max_encoder_length=56,                   # look back 56 days
        max_prediction_length=HORIZON,           # predict 28 days
        static_categoricals=["store_id", "item_id"],
        time_varying_known_categoricals=["dow_s", "month_s"],
        time_varying_known_reals=["time_idx", "sell_price", "snap", "promo", "is_event"],
        time_varying_unknown_reals=["sales"],
        target_normalizer=GroupNormalizer(groups=["id"], transformation="softplus"),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )
    validation = TimeSeriesDataSet.from_dataset(
        training, df, predict=True, stop_randomization=True)

    train_loader = training.to_dataloader(train=True, batch_size=128, num_workers=0)
    val_loader = validation.to_dataloader(train=False, batch_size=256, num_workers=0)

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.01,
        hidden_size=16,
        attention_head_size=2,
        dropout=0.1,
        hidden_continuous_size=8,
        loss=QuantileLoss([0.05, 0.25, 0.5, 0.75, 0.95]),
    )
    trainer = pl.Trainer(max_epochs=epochs, accelerator="auto", gradient_clip_val=0.1)
    trainer.fit(tft, train_loader, val_loader)
    trainer.save_checkpoint(MODELS / "tft.ckpt")

    # Predict the last 28 days and turn the output into a tidy table
    out = tft.predict(val_loader, mode="prediction", return_index=True)
    preds = pd.DataFrame(out.output.cpu().numpy())          # one row per series, 28 columns
    preds["id"] = out.index["id"].values
    preds["start"] = out.index["time_idx"].values
    preds = preds.melt(id_vars=["id", "start"], var_name="h", value_name="tft")
    preds["time_idx"] = preds["start"] + preds["h"]
    preds["tft"] = preds["tft"].clip(lower=0)

    merged = df.merge(preds[["id", "time_idx", "tft"]], on=["id", "time_idx"])
    print("rows scored:", len(merged), "(expect 5600)")
    print("TFT (WAPE, RMSE):", score(merged["sales"], merged["tft"]))
    merged[["id", "date", "tft"]].to_csv(OUTPUTS / "tft_preds.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    main(args.epochs)