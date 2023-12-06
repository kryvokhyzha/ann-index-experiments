import json
from pathlib import Path
from typing import Any, Tuple

import lightning.pytorch as pl
import numpy as np
import open_clip
import pandas as pd
import torch
import torch.utils.data


def load_dataframe(path: str | Path) -> pd.DataFrame:
    """Load a dataframe from a json file.

    Parameters
    ----------
    path : str or Path
        Path to the json file.

    Returns
    -------
    pd.DataFrame
        Dataframe with columns "intents" and "targets".
    """
    with open(path, "r") as f:
        data = json.load(f)

    dataset_list = [pd.DataFrame(data[key], columns=["intent", "target"]) for key in data.keys()]

    return pd.concat(dataset_list, axis=0, sort=False).reset_index(drop=True)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: Any):
        """Dataset class for the intent classification task.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe with columns "intent" and "target".
        tokenizer : Any
            Tokenizer to use.
        """
        super().__init__()
        self.df = df
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return self.df.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        intent, target = self.df.iloc[idx]
        token_ids = self.tokenizer(intent).squeeze(0)

        return token_ids, intent, target


class InferenceModule(pl.LightningModule):
    def __init__(self, text_model: Any):
        super().__init__()
        self.text_model = text_model

        self.random_projection_128 = torch.nn.LazyLinear(out_features=128)
        self.random_projection_64 = torch.nn.LazyLinear(out_features=64)
        self.random_projection_32 = torch.nn.LazyLinear(out_features=32)
        self.random_projection_16 = torch.nn.LazyLinear(out_features=16)
        self.random_projection_8 = torch.nn.LazyLinear(out_features=8)

    def forward(self, token_ids: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        embeddings = self.text_model(token_ids)
        embeddings_128 = self.random_projection_128(embeddings)
        embeddings_64 = self.random_projection_64(embeddings)
        embeddings_32 = self.random_projection_32(embeddings)
        embeddings_16 = self.random_projection_16(embeddings)
        embeddings_8 = self.random_projection_8(embeddings)

        return embeddings, embeddings_128, embeddings_64, embeddings_32, embeddings_16, embeddings_8

    def predict_step(self, batch: Tuple[torch.Tensor, ...]) -> Any:
        token_ids, intent, target = batch
        embeddings = tuple(map(lambda x: x.detach().cpu().numpy().astype(np.float32), self(token_ids)))
        return *embeddings, intent, target


def process_data(
    tokenizer: Any,
    model: torch.nn.Module,
    df: pd.DataFrame,
    output_file_path: str | Path,
    batch_size: int = 512,
    num_workers: int = 8,
):
    dataset = Dataset(tokenizer=tokenizer, df=df)
    pl_module = InferenceModule(text_model=model)
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=True,
        shuffle=False,
    )

    trainer = pl.Trainer(
        accelerator="auto",
        precision="16-mixed",
    )
    result = trainer.predict(model=pl_module, dataloaders=dataloader)

    total = []
    for batch in result:
        total.extend(zip(*batch))

    pd.DataFrame(
        data=total, columns=["embedding", "proj128", "proj64", "proj32", "proj16", "proj8", "intent", "target"]
    ).to_parquet(output_file_path, index=False, engine="pyarrow")


def main():
    path_to_root = Path(__file__).parent.parent
    path_to_data = path_to_root / "data"

    df_real = load_dataframe(path_to_data / "data_full.json")
    df_synthetic = pd.DataFrame([(str(i), "synth") for i in range(25_000)], columns=["intent", "target"])

    model, _, _ = open_clip.create_model_and_transforms(
        model_name="xlm-roberta-base-ViT-B-32",
        pretrained="laion5b_s13b_b90k",
        force_custom_text=True,
    )

    tokenizer = open_clip.get_tokenizer(
        model_name="xlm-roberta-base-ViT-B-32",
    )

    print("Processing real data...")
    process_data(
        tokenizer=tokenizer,
        model=model.text,
        df=df_real,
        output_file_path=str(path_to_data / "xlm-roberta-embeddings.parquet"),
    )
    print("Done.")

    print("Processing synthetic data...")
    process_data(
        tokenizer=tokenizer,
        model=model.text,
        df=df_synthetic,
        output_file_path=str(path_to_data / "xlm-roberta-synthetic-embeddings.parquet"),
    )
    print("Done.")


if __name__ == "__main__":
    main()
