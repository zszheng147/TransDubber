from torch.utils.data import DataLoader
import lightning as L

from dataset import (
    ASROnlyPauseDataset,
    ASTTextDataset,
    DynamicBatchSampler,
    MixedSFTDataset,
    Qwen3ASRCollator,
    T2STDataset,
)


# ------------------------ Define the data module ------------------------ #
class LitDataModule(L.LightningDataModule):
    def __init__(self, config, processor=None):
        super().__init__()
        self.config = config
        self.processor = processor
        self.collator = Qwen3ASRCollator(
            processor=processor,
            sampling_rate=config.sampling_rate,
        )

    def setup(self, stage: str):
        if stage == "fit":
            base_pause_span_source = (
                "inserted"
                if bool(getattr(
                    self.config,
                    "enable_base_pause_speech_supervision",
                    False,
                ))
                else "none"
            )
            train_datasets = []
            train_repeats = []
            train_parts = []
            if list(getattr(self.config, "train_parquet_dir", []) or []):
                full_train = T2STDataset(
                    self.config,
                    "train",
                    processor=self.processor,
                    target_pause_span_source=base_pause_span_source,
                )
                train_datasets.append(full_train)
                train_repeats.append(1)
                train_parts.append(f"base={len(full_train)}")

            paired_paths = list(
                getattr(
                    self.config,
                    "paired_pause_train_parquet_dir",
                    [],
                )
                or []
            )
            if paired_paths:
                paired_train = T2STDataset(
                    self.config,
                    "train",
                    processor=self.processor,
                    paths=paired_paths,
                    target_pause_span_source=getattr(
                        self.config, "paired_pause_span_source", "audited",
                    ),
                )
                paired_repeat = int(getattr(
                    self.config, "paired_pause_repeat", 1,
                ))
                if paired_repeat < 1:
                    raise ValueError("paired_pause_repeat must be >= 1")
                train_datasets.append(paired_train)
                train_repeats.append(paired_repeat)
                train_parts.append(
                    f"paired_pause={len(paired_train)}x{paired_repeat}"
                )

            ast_paths = list(
                getattr(self.config, "ast_text_train_parquet_dir", []) or []
            )
            if ast_paths:
                ast_train = ASTTextDataset(
                    self.config, "train", processor=self.processor,
                )
                ast_repeat = int(getattr(self.config, "ast_text_repeat", 1))
                if ast_repeat < 1:
                    raise ValueError("ast_text_repeat must be >= 1")
                train_datasets.append(ast_train)
                train_repeats.append(ast_repeat)
                train_parts.append(f"ast_text={len(ast_train)}x{ast_repeat}")

            asr_paths = list(
                getattr(self.config, "asr_only_train_parquet_dir", []) or []
            )
            if asr_paths:
                asr_train = ASROnlyPauseDataset(
                    self.config, "train", processor=self.processor,
                )
                repeat = int(getattr(self.config, "asr_only_repeat", 1))
                if repeat < 1:
                    raise ValueError("asr_only_repeat must be >= 1")
                train_datasets.append(asr_train)
                train_repeats.append(repeat)
                train_parts.append(f"asr_only={len(asr_train)}x{repeat}")

            if not train_datasets:
                raise ValueError(
                    "No training data configured: set at least one of "
                    "train_parquet_dir, paired_pause_train_parquet_dir, "
                    "ast_text_train_parquet_dir, asr_only_train_parquet_dir."
                )

            if len(train_datasets) > 1:
                self.train_dataset = MixedSFTDataset(
                    train_datasets, repeats=train_repeats,
                )
            else:
                self.train_dataset = train_datasets[0]
            print(
                "SFT train data: "
                + ", ".join(train_parts)
                + f", total={len(self.train_dataset)}, "
                + f"base_pause_regions={base_pause_span_source == 'inserted'}"
            )

            val_datasets = []
            if list(getattr(self.config, "dev_parquet_dir", []) or []):
                val_datasets.append(T2STDataset(
                    self.config,
                    "dev",
                    processor=self.processor,
                    target_pause_span_source=base_pause_span_source,
                ))
            paired_dev_paths = list(getattr(
                self.config, "paired_pause_dev_parquet_dir", [],
            ) or [])
            if paired_dev_paths:
                val_datasets.append(T2STDataset(
                    self.config,
                    "dev",
                    processor=self.processor,
                    paths=paired_dev_paths,
                    target_pause_span_source=getattr(
                        self.config, "paired_pause_span_source", "audited",
                    ),
                ))
            if list(getattr(self.config, "ast_text_dev_parquet_dir", []) or []):
                val_datasets.append(ASTTextDataset(
                    self.config, "dev", processor=self.processor,
                ))
            if list(getattr(self.config, "asr_only_dev_parquet_dir", []) or []):
                val_datasets.append(ASROnlyPauseDataset(
                    self.config, "dev", processor=self.processor,
                ))
            if not val_datasets:
                raise ValueError(
                    "No dev data configured: set at least one of "
                    "dev_parquet_dir, ast_text_dev_parquet_dir, "
                    "asr_only_dev_parquet_dir."
                )
            self.val_dataset = (
                MixedSFTDataset(val_datasets)
                if len(val_datasets) > 1
                else val_datasets[0]
            )
        elif stage == "validate":
            base_pause_span_source = (
                "inserted"
                if bool(getattr(
                    self.config,
                    "enable_base_pause_speech_supervision",
                    False,
                ))
                else "none"
            )
            val_datasets = []
            if list(getattr(self.config, "dev_parquet_dir", []) or []):
                val_datasets.append(T2STDataset(
                    self.config,
                    "dev",
                    processor=self.processor,
                    target_pause_span_source=base_pause_span_source,
                ))
            paired_dev_paths = list(getattr(
                self.config, "paired_pause_dev_parquet_dir", [],
            ) or [])
            if paired_dev_paths:
                val_datasets.append(T2STDataset(
                    self.config,
                    "dev",
                    processor=self.processor,
                    paths=paired_dev_paths,
                    target_pause_span_source=getattr(
                        self.config, "paired_pause_span_source", "audited",
                    ),
                ))
            if list(getattr(self.config, "ast_text_dev_parquet_dir", []) or []):
                val_datasets.append(ASTTextDataset(
                    self.config, "dev", processor=self.processor,
                ))
            if list(getattr(self.config, "asr_only_dev_parquet_dir", []) or []):
                val_datasets.append(ASROnlyPauseDataset(
                    self.config, "dev", processor=self.processor,
                ))
            if not val_datasets:
                raise ValueError(
                    "No dev data configured: set at least one of "
                    "dev_parquet_dir, ast_text_dev_parquet_dir, "
                    "asr_only_dev_parquet_dir."
                )
            self.val_dataset = (
                MixedSFTDataset(val_datasets)
                if len(val_datasets) > 1
                else val_datasets[0]
            )

    def train_dataloader(self):
        # Use dynamic batch sampler to group samples by audio duration
        sampler = DynamicBatchSampler(
            dataset=self.train_dataset,
            default_batch_size=self.config.batch_size,
            boundaries=getattr(self.config, 'dynamic_batch_boundaries', None),
            bucket_batch_sizes=getattr(self.config, 'dynamic_bucket_batch_sizes', None),
            shuffle=True,
            drop_last=True,
            seed=getattr(self.config, 'seed', 42),
        )
        return DataLoader(
            self.train_dataset,
            batch_sampler=sampler,
            collate_fn=self.collator,
            num_workers=self.config.num_workers,
            pin_memory=True,
            persistent_workers=self.config.num_workers > 0,
            prefetch_factor=self.config.prefetch_factor if self.config.num_workers > 0 else None,
        )

    def val_dataloader(self):
        sampler = DynamicBatchSampler(
            dataset=self.val_dataset,
            default_batch_size=self.config.batch_size,
            boundaries=getattr(self.config, 'dynamic_batch_boundaries', None),
            bucket_batch_sizes=getattr(self.config, 'dynamic_bucket_batch_sizes', None),
            shuffle=False,
            # Keep the full validation set. DDP may repeat a small number of
            # batches to equalize the number of batches on each rank.
            drop_last=False,
            seed=getattr(self.config, 'seed', 42),
        )
        return DataLoader(
            self.val_dataset,
            batch_sampler=sampler,
            collate_fn=self.collator,
            num_workers=self.config.num_workers,
            pin_memory=True,
            persistent_workers=self.config.num_workers > 0,
            prefetch_factor=self.config.prefetch_factor if self.config.num_workers > 0 else None,
        )
