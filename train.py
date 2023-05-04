import pytorch_lightning as pl

from dataset import dataset_meta, CSVDataModule
from model import Inception_V3

if __name__ == "__main__":
    meta = dataset_meta("data/argument.csv")
    total_len = sum([meta[str(idx)] for idx in range(3)])
    model = Inception_V3(data_len_list=[meta[str(idx)] for idx in range(3)])
    # 实例化DataModule类
    data = CSVDataModule(csv_fp="data/argument.csv", npy_fp="data/argument.npy", batch_size=16)

    # 定义训练器
    trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=100, check_val_every_n_epoch=5,
                         auto_lr_find=True, auto_scale_batch_size=True)

    trainer.fit(model, data)
