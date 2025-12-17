#!/usr/bin/env python3

import logging
import os
import random
import time

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.dataset import AudioFeatDataset, MPerClassSampler
from src.eval_testset import eval_for_map_with_feat
from src.pytorch_utils import get_lr, scan_and_load_checkpoint
from src.scheduler import UserDefineExponentialLR

# setting this to False in Apple Silicon context showed negligible impact.
torch.backends.cudnn.benchmark = True

# ALL_TEST_SETS defines recognized testset names. A testset is active when its
# name appears both here AND as a key in hparams.yaml with query_path/ref_path.
#
# For mAP-based early stopping, testsets are partitioned into two tiers:
# - Stopping testsets (hp["map_stopping_testsets"]): Drive stopping/selection decisions
# - Benchmark testsets (all others): Logged only, never influence training
#
# This separation enables research-grade evaluation where benchmark results
# remain uncontaminated by training decisions.
#
# Original CoverHunter only included a configuration for "covers80"
# but also listed "shs_test", "dacaos" (presumably a typo for da-tacos), "hymf_20", "hymf_100"
ALL_TEST_SETS = ["covers80", "shs100k-test", "reels50easy", "reels50hard", "reels50transpose", "meertens80disjoint", "meertens100", "meertens50hard", "meertens50easy", "meertens50"]


class Trainer:
    def __init__(
        self,
        hp,
        model,
        device,
        log_path,
        checkpoint_dir,
        model_dir,
        only_eval,
        first_eval,
    ):
        """
        Trainer class to organize the training methods.

        Supports two early stopping modes controlled by hp["early_stop_metric"]:
        - "val_loss" (default): Stop when validation loss plateaus. All testsets
          remain uncontaminated benchmarks suitable for publication.
        - "mAP": Stop when smoothed mAP plateaus on hp["map_stopping_testsets"].
          Testsets not in that list are benchmark-only (logged but never
          influence training decisions), preserving research-grade separation.

        Args:
        ----
          hp: dict
            The hyperparameters as a dict.
          model: Model
            The model class that will be used.
          device: torch.device
            The device that will be used for computation.
          log_path: str
            TensorBoard SummaryWriter log path.
          checkpoint_dir: str
            Directory for saving/loading model checkpoints.
          model_dir: str
            Base directory for the training run.
          only_eval: bool
            If True, run evaluation only (no training).
          first_eval: bool
            If True, evaluate before first training epoch.

        Attributes (early stopping state):
            best_validation_loss: float
                Lowest validation loss observed (val_loss mode).
            early_stopping_counter: int
                Epochs since val_loss improvement.
            map_stopping_testsets: list[str]
                Testsets that drive mAP-based stopping decisions.
            best_raw_map: float
                Highest raw mAP observed on stopping testsets.
            best_raw_map_epoch: int or None
                Epoch where best_raw_map was achieved (for checkpoint selection).
            smoothed_map: float or None
                EMA-smoothed mAP for stopping decisions.
            map_stopping_counter: int
                Epochs since smoothed mAP improvement.

        """
        self.hp = hp
        self.model = model(hp).to(device)
        self.device = device
        self.model_dir = model_dir
        self.checkpoint_dir = checkpoint_dir
        self.logger = logging.getLogger("Trainer")
        self.only_eval = only_eval
        self.first_eval = first_eval
        self.best_validation_loss = float("inf")
        self.early_stopping_counter = 0

        # --- mAP-based early stopping state (used when early_stop_metric="mAP") ---
        # Testsets in this list influence stopping/selection decisions.
        # Testsets NOT in this list are benchmark-only (logged but never drive decisions).
        self.map_stopping_testsets = [
            t for t in hp.get("map_stopping_testsets", []) if t in hp
        ]
        self.best_smoothed_map = 0.0
        self.best_raw_map = 0.0
        self.best_raw_map_epoch = None
        self.smoothed_map = None  # None until first observation
        self.map_stopping_counter = 0
        # Cache of most recent mAP scores by testset (for stopping calculation)
        self._current_epoch_map_scores = {}

        self.test_sets = [d for d in ALL_TEST_SETS if d in hp]

        self.training_data = []
        infer_len = hp["chunk_frame"][0] * hp["mean_size"]

        for chunk_len in hp["chunk_frame"]:
            self.training_data.append(
                DataLoader(
                    AudioFeatDataset(
                        hp,
                        hp["train_path"],
                        train=True,
                        mode=hp["mode"],
                        chunk_len=chunk_len * hp["mean_size"],
                        logger=self.logger,
                    ),
                    num_workers=hp["num_workers"],
                    shuffle=False,
                    sampler=MPerClassSampler(
                        data_path=hp["train_path"],
                        m=hp["m_per_class"],
                        batch_size=hp["batch_size"],
                        distribute=False,
                        logger=self.logger,
                        seed=hp["seed"],
                    ),
                    batch_size=hp["batch_size"],
                    pin_memory=True,
                    drop_last=True,
                )
            )

        # At inference stage, we only use chunk with fixed length
        self.logger.info("Init val and test data loader")
        self.sample_training_data = None
        if "val_path" in hp:
            self.sample_training_data = DataLoader(
                AudioFeatDataset(
                    hp,
                    hp["val_path"],
                    train=False,
                    chunk_len=infer_len,
                    mode=hp["mode"],
                    logger=self.logger,
                ),
                num_workers=1,
                shuffle=False,
                sampler=MPerClassSampler(
                    data_path=hp["val_path"],
                    # m=hp["m_per_class"],
                    m=1,
                    batch_size=hp["batch_size"],
                    distribute=False,
                    logger=self.logger,
                    seed=hp["seed"],
                ),
                batch_size=hp["batch_size"],
                pin_memory=True,
                collate_fn=None,
                drop_last=False,
            )

        self.test_data = None
        if "test_path" in hp:
            self.test_data = DataLoader(
                AudioFeatDataset(
                    hp,
                    hp["test_path"],
                    chunk_len=infer_len,
                    mode=hp["mode"],
                    logger=self.logger,
                ),
                num_workers=1,
                shuffle=False,
                sampler=MPerClassSampler(
                    data_path=hp["test_path"],
                    m=hp["m_per_class"],
                    batch_size=hp["batch_size"],
                    distribute=False,
                    logger=self.logger,
                    seed=hp["seed"],
                ),
                batch_size=hp["batch_size"],
                pin_memory=True,
                collate_fn=None,
                drop_last=False,
            )

        self.epoch = -1
        self.step = 1

        self.summary_writer = None
        os.makedirs(log_path, exist_ok=True)
        if not only_eval:
            self.summary_writer = SummaryWriter(log_path)

    def configure_optimizer(self):
        """
        Configure the model optimizer.
        """
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            self.hp["learning_rate"],
            betas=[self.hp["adam_b1"], self.hp["adam_b2"]],
        )

    def configure_scheduler(self):
        """
        Configure the model scheduler.
        """
        self.scheduler = UserDefineExponentialLR(
            self.optimizer,
            gamma=self.hp["lr_decay"],
            min_lr=self.hp["min_lr"],
            last_epoch=self.epoch,
        )

    def load_model(self, advanced=False):
        """
        Load the current model from checkpoint_dir.
        """
        self.step, self.epoch = load_checkpoint(
            self.model,
            self.optimizer,
            self.checkpoint_dir,
            advanced=advanced,
        )

    def reset_learning_rate(self, new_lr=None):
        """
        Utility to allow custom control of learning rate during training,
        after loading a model from a checkpoint that would otherwise inherit
        the learning rate from the previous epoch via the checkpoint.

        Args:
        ----
        new_lr : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        ----
        None.

        """
        if new_lr is None:
            new_lr = self.hp["learning_rate"]
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr
        # Recreate the scheduler with the new learning rate
        self.configure_scheduler()

    def save_model(self):
        """
        Save the current model to checkpoint_dir.
        """
        if self.epoch % self.hp.get("every_n_epoch_to_save", 1) != 0:
            return

        save_checkpoint(
            self.model,
            self.optimizer,
            self.step,
            self.epoch,
            self.checkpoint_dir,
        )

    def train_epoch(self, epoch, first_eval):
        """
        Train for the given epoch.

        Skip it if first_eval is set.
        """
        if first_eval:
            return

        train_step = None
        start = time.time()
        self.epoch = epoch
        self.logger.info("Start to train for epoch %d", self.epoch)
        self.step = train_one_epoch(
            self.model,
            self.optimizer,
            self.scheduler,
            self.training_data,
            self.step,
            train_step=train_step,
            device=self.device,
            sw=self.summary_writer,
            logger=self.logger,
        )
        self.logger.info(
            "Time for train epoch %d step %d is %.1fs",
            self.epoch,
            self.step,
            time.time() - start,
        )

    def validate_one(self, data_type):
        """
        Compute validation or test loss for the current epoch.

        Evaluates the model on the specified data loader and logs loss metrics.
        For data_type="val", also updates val_loss early stopping state
        (best_validation_loss, early_stopping_counter) when early_stop_metric
        is "val_loss".

        Args:
            data_type: "val" or "test"
                Which data loader to evaluate.

        Note:
            Respects hp["every_n_epoch_to_test"] to skip evaluation on
            intermediate epochs. This is separate from testset mAP evaluation,
            which occurs in eval_and_log().
        """
        if not self.epoch % self.hp.get("every_n_epoch_to_test", 1) == 0:
            return

        start = time.time()

        if data_type == "val":
            data = self.sample_training_data
        elif data_type == "test":
            data = self.test_data

        if not data:
            return

        self.logger.info("compute %s at epoch-%d", data_type, self.epoch)

        res = validate(
            self.model,
            data,
            data_type,
            epoch_num=self.epoch,
            device=self.device,
            sw=self.summary_writer,
            logger=self.logger,
        )
        validation_loss = res["foc_loss"] / res["count"]
        self.logger.info(
            "count:%d, avg_foc_loss:%f", res["count"], validation_loss
        )

        self.logger.info(
            "Time for %s is %.1fs\n", data_type, time.time() - start
        )

        if data_type == "val":
            if validation_loss < self.best_validation_loss:
                self.best_validation_loss = validation_loss
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1

    def eval_and_log(self):
        """
        Run validation, evaluate testsets, log results, and update early stopping state.

        Performs three categories of evaluation:
        1. Validation loss on val_path data (via validate_one)
        2. Test loss on test_path data if configured (via validate_one)
        3. mAP on each configured testset (logged to TensorBoard)

        For mAP-based early stopping (early_stop_metric="mAP"):
        - Collects mAP scores from all evaluated testsets
        - Filters to map_stopping_testsets for stopping decisions
        - Updates smoothed mAP and stopping counter via _check_map_early_stopping()
        - Testsets not in map_stopping_testsets are logged as benchmarks only

        Testset evaluation respects per-testset every_n_epoch_to_test settings.
        """
        self.validate_one("val")
        self.validate_one("test")

        # Clear previous epoch's mAP cache
        self._current_epoch_map_scores = {}

        valid_testlist = []
        for testset_name in self.test_sets:
            hp_test = self.hp[testset_name]
            if self.epoch % hp_test.get("every_n_epoch_to_test", 1) == 0:
                valid_testlist.append(testset_name)

        for testset_name in valid_testlist:
            hp_test = self.hp[testset_name]
            self.logger.info(
                "Compute %s at epoch: %s", testset_name, self.epoch
            )

            start = time.time()
            save_name = hp_test.get("save_name", testset_name)
            embed_dir = os.path.join(
                self.model_dir, f"embed_{self.epoch}_{save_name}"
            )
            query_in_ref_path = hp_test.get("query_in_ref_path", None)
            mean_ap, hit_rate, _ = eval_for_map_with_feat(
                self.hp,
                self.model,
                embed_dir,
                query_path=hp_test["query_path"],
                ref_path=hp_test["ref_path"],
                query_in_ref_path=query_in_ref_path,
                batch_size=self.hp["batch_size"],
                device=self.device,
                logger=self.logger,
            )

            self.summary_writer.add_scalar(
                f"mAP/{testset_name}", mean_ap, self.epoch
            )
            self.summary_writer.add_scalar(
                f"hit_rate/{testset_name}", hit_rate, self.epoch
            )
            self.logger.info(
                "Test %s, hit_rate:%s, map:%s", testset_name, hit_rate, mean_ap
            )
            self.logger.info(
                "Time for test-%s is %d sec\n",
                testset_name,
                int(time.time() - start),
            )

            # Cache mAP for early stopping check
            self._current_epoch_map_scores[testset_name] = mean_ap

        # Perform mAP-based early stopping check if configured
        if (
            self.hp.get("early_stop_metric") == "mAP"
            and self.map_stopping_testsets
        ):
            self._check_map_early_stopping()

    def aggregate_stopping_map_scores(
        self, map_scores: dict[str, float]
    ) -> float:
        """
        Aggregate mAP scores across stopping-relevant testsets.

        This method is intentionally isolated to make it easy for users to
        customize the aggregation strategy for their research needs.

        Args:
            map_scores: Dictionary mapping testset names to their mAP values.
                        Only includes testsets from map_stopping_testsets.

        Returns:
            Aggregated mAP score used for early stopping decisions.

        Current implementation: arithmetic mean.

        Alternative implementations users might substitute:
            - Weighted mean:
                weights = {"reels50hard": 0.5, "reels50easy": 0.3, ...}
                return sum(scores[k] * weights[k] for k in scores) /
                    sum(weights.values())
            - Worst-case (conservative):
                return min(scores.values())
            - Primary testset only:
                return scores.get("reels50hard", 0.0)
        """
        if not map_scores:
            return 0.0
        return sum(map_scores.values()) / len(map_scores)

    def _update_smoothed_map(self, raw_map: float) -> float:
        """
        Update exponential moving average (EMA) of mAP for early stopping.

        The EMA filters noise in mAP measurements (especially on small testsets)
        to prevent false early stopping triggers while remaining responsive to
        genuine performance plateaus.

        Args:
            raw_map: Current epoch's aggregated mAP score.

        Returns:
            Updated smoothed mAP value.
        """
        alpha = self.hp.get("map_smoothing_alpha", 0.3)
        if self.smoothed_map is None:
            self.smoothed_map = raw_map
        else:
            self.smoothed_map = (
                alpha * raw_map + (1 - alpha) * self.smoothed_map
            )
        return self.smoothed_map

    def _check_map_early_stopping(self) -> None:
        """
        Evaluate early stopping based on smoothed mAP from stopping testsets.

        Called after testset evaluation when early_stop_metric="mAP".
        Updates best_raw_map, best_raw_map_epoch, and map_stopping_counter.

        The stopping decision uses smoothed mAP to filter noise, but
        best_raw_map_epoch tracks the actual peak for checkpoint selection.
        """
        if not self._current_epoch_map_scores:
            return

        # Filter to only stopping-relevant testsets
        stopping_scores = {
            k: v
            for k, v in self._current_epoch_map_scores.items()
            if k in self.map_stopping_testsets
        }

        if not stopping_scores:
            # No stopping testsets were evaluated this epoch
            return

        raw_map = self.aggregate_stopping_map_scores(stopping_scores)
        smoothed = self._update_smoothed_map(raw_map)

        # Track best raw mAP for checkpoint selection
        if raw_map > self.best_raw_map:
            self.best_raw_map = raw_map
            self.best_raw_map_epoch = self.epoch
            self.logger.info(
                f"New best raw mAP: {raw_map:.4f} at epoch {self.epoch}"
            )

        # Stopping decision based on smoothed mAP
        if smoothed > self.best_smoothed_map:
            self.best_smoothed_map = smoothed
            self.map_stopping_counter = 0
        else:
            self.map_stopping_counter += 1

        self.logger.info(
            f"mAP stopping check: raw={raw_map:.4f}, smoothed={smoothed:.4f}, "
            f"best_smoothed={self.best_smoothed_map:.4f}, "
            f"patience={self.map_stopping_counter}/{self.hp.get('map_stopping_patience', 5)}"
        )

    def train(self, max_epochs):
        """
        Train the model with configurable early stopping up to max_epochs.

        Runs the training loop until max_epochs or early stopping triggers.
        Early stopping behavior depends on hp["early_stop_metric"]:

        Mode "val_loss" (default):
            Stops when validation loss has not improved for
            hp["early_stopping_patience"] epochs. All testsets remain
            uncontaminated and suitable for publication as benchmarks.

        Mode "mAP":
            Stops when smoothed mAP on hp["map_stopping_testsets"] has not
            improved for hp["map_stopping_patience"] epochs. Checkpoint
            selection uses best_raw_map_epoch (actual peak, not smoothed).
            Testsets not in map_stopping_testsets are evaluated and logged
            but never influence stopping, preserving their validity as
            held-out benchmarks for research publication.

            For research with publication-grade benchmarks in this mode,
            include only validation-tier testsets in map_stopping_testsets.
            Testsets not in that list remain uncontaminated benchmarks.

        Args:
            max_epochs: Maximum number of epochs to train.

        Returns:
            None. Updates instance state including:
            - self.epoch: Final epoch number
            - self.best_validation_loss: Best val loss (val_loss mode)
            - self.best_raw_map: Best mAP achieved (mAP mode)
            - self.best_raw_map_epoch: Epoch of best mAP (mAP mode)

        Note:
            When every_n_epoch_to_test > 1, forces a final mAP evaluation
            to ensure accurate metrics for train_tune experiment summaries.

        """
        early_stop_metric = self.hp.get("early_stop_metric", "val_loss")
        val_loss_patience = self.hp.get("early_stopping_patience", 1000)
        map_patience = self.hp.get("map_stopping_patience", 5)

        # Validate mAP mode configuration
        if early_stop_metric == "mAP" and not self.map_stopping_testsets:
            self.logger.warning(
                "early_stop_metric='mAP' but no valid map_stopping_testsets found. "
                "Falling back to val_loss mode."
            )
            early_stop_metric = "val_loss"

        if early_stop_metric == "mAP":
            self.logger.info(
                f"Using mAP-based early stopping with testsets: {self.map_stopping_testsets}"
            )
            benchmark_testsets = [
                t
                for t in self.test_sets
                if t not in self.map_stopping_testsets
            ]
            if benchmark_testsets:
                self.logger.info(
                    f"Benchmark testsets (not used for stopping): {benchmark_testsets}"
                )

        first_eval = self.first_eval
        for epoch in range(max(0, 1 + self.epoch), max_epochs):
            self.train_epoch(epoch, first_eval)
            self.eval_and_log()
            self.save_model()

            # Check early stopping based on configured metric
            should_stop = False
            if early_stop_metric == "val_loss":
                if self.early_stopping_counter >= val_loss_patience:
                    self.logger.info(
                        "Early stopping at epoch %d due to val_loss plateau "
                        "(no improvement for %d epochs)",
                        self.epoch,
                        val_loss_patience,
                    )
                    should_stop = True
            elif early_stop_metric == "mAP":
                if self.map_stopping_counter >= map_patience:
                    self.logger.info(
                        "Early stopping at epoch %d due to smoothed mAP plateau "
                        "(no improvement for %d epochs). "
                        "Best raw mAP %.4f at epoch %d.",
                        self.epoch,
                        map_patience,
                        self.best_raw_map,
                        self.best_raw_map_epoch,
                    )
                    should_stop = True

            if should_stop or self.only_eval:
                self._ensure_final_evaluation()
                return

            first_eval = False

        # Training completed without early stopping
        self._ensure_final_evaluation()

    def _ensure_final_evaluation(self):
        """
        Ensure final epoch has mAP evaluation for accurate reporting.

        When every_n_epoch_to_test > 1, the final epoch may not have been
        evaluated. This forces evaluation so train_tune can report accurate
        final mAP in experiment summaries.
        """
        every_n = self.hp.get("every_n_epoch_to_test", 1)
        if every_n > 1 and self.epoch % every_n != 0:
            self.logger.info(
                f"Forcing final mAP evaluation at epoch {self.epoch} "
                f"(every_n_epoch_to_test={every_n})"
            )
            # Temporarily override to force evaluation
            original_every_n = {}
            for testset_name in self.test_sets:
                hp_test = self.hp[testset_name]
                original_every_n[testset_name] = hp_test.get(
                    "every_n_epoch_to_test", 1
                )
                hp_test["every_n_epoch_to_test"] = 1

            # Re-run testset evaluation only (skip val/test loaders)
            self._current_epoch_map_scores = {}
            for testset_name in self.test_sets:
                hp_test = self.hp[testset_name]
                save_name = hp_test.get("save_name", testset_name)
                embed_dir = os.path.join(
                    self.model_dir, f"embed_{self.epoch}_{save_name}"
                )
                query_in_ref_path = hp_test.get("query_in_ref_path", None)
                mean_ap, hit_rate, _ = eval_for_map_with_feat(
                    self.hp,
                    self.model,
                    embed_dir,
                    query_path=hp_test["query_path"],
                    ref_path=hp_test["ref_path"],
                    query_in_ref_path=query_in_ref_path,
                    batch_size=self.hp["batch_size"],
                    device=self.device,
                    logger=self.logger,
                )
                self.summary_writer.add_scalar(
                    f"mAP/{testset_name}", mean_ap, self.epoch
                )
                self.summary_writer.add_scalar(
                    f"hit_rate/{testset_name}", hit_rate, self.epoch
                )
                self._current_epoch_map_scores[testset_name] = mean_ap

            # Restore original settings
            for testset_name, orig_val in original_every_n.items():
                self.hp[testset_name]["every_n_epoch_to_test"] = orig_val


def save_checkpoint(model, optimizer, step, epoch, checkpoint_dir) -> None:
    g_checkpoint_path = f"{checkpoint_dir}/g_{epoch:08d}"

    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    elif isinstance(model, torch.nn.parallel.DistributedDataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()

    torch.save({"generator": state_dict}, g_checkpoint_path)
    d_checkpoint_path = f"{checkpoint_dir}/do_{epoch:08d}"
    torch.save(
        {"optim_g": optimizer.state_dict(), "steps": step, "epoch": epoch},
        d_checkpoint_path,
    )
    logging.info(f"save checkpoint to {g_checkpoint_path}")
    logging.info(f"save step:{step}, epoch:{epoch}")


def load_checkpoint(
    model, optimizer=None, checkpoint_dir=None, advanced=False
):
    state_dict_g = scan_and_load_checkpoint(checkpoint_dir, "g_")
    state_dict_do = scan_and_load_checkpoint(checkpoint_dir, "do_")
    if state_dict_g:
        if advanced:
            model_dict = model.state_dict()
            valid_dict = {
                k: v for k, v in state_dict_g.items() if k in model_dict
            }
            model_dict.update(valid_dict)
            model.load_state_dict(model_dict)
            for k in model_dict:
                if k not in state_dict_g:
                    logging.warning(f"{k} not be initialized")
        else:
            model.load_state_dict(state_dict_g["generator"])
            # self.load_state_dict(state_dict_g)

        logging.info(f"load g-model from {checkpoint_dir}")

    if state_dict_do is None:
        logging.info("using init value of steps and epoch")
        step, epoch = 1, -1
    else:
        step, epoch = state_dict_do["steps"] + 1, state_dict_do["epoch"]
        logging.info(f"load d-model from {checkpoint_dir}")
        optimizer.load_state_dict(state_dict_do["optim_g"])

    logging.info(f"step:{step}, epoch:{epoch}")
    return step, epoch


def train_one_epoch(
    model,
    optimizer,
    scheduler,
    train_loader_lst,
    step,
    train_step=None,
    device="mps",
    sw=None,
    logger=None,
):
    """train one epoch with multi data_loader"""
    init_step = step
    model.train()  # torch.nn.Module.train sets model in training mode
    idx_loader = list(range(len(train_loader_lst)))
    for batch_lst in zip(*train_loader_lst):
        random.shuffle(idx_loader)
        for idx in idx_loader:
            batch = list(batch_lst)[idx]
            if step % 1000 == 0:
                scheduler.step()
            model.train()
            _, feat, label = batch
            feat = batch[1].float().to(device)
            label = batch[2].long().to(device)

            optimizer.zero_grad()
            total_loss, losses = model.compute_loss(feat, label)

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            _loss_memory = {"lr": get_lr(optimizer)}
            for key, value in losses.items():
                _loss_memory.update({key: value.item()})
            _loss_memory.update({"total": total_loss.item()})

            if step == 1 or step % 100 == 0:
                log_info = f"Steps:{step:d}"
                for k, v in _loss_memory.items():
                    if k == "lr":
                        log_info += f" lr:{v:.6f}"
                    else:
                        log_info += f" {k}:{v:.3f}"
                    if sw:
                        sw.add_scalar(f"csi/{k}", v, step)
                if logger:
                    logger.info(log_info)
            step += 1

            if train_step is not None:
                if (step - init_step) == train_step:
                    return step
    return step


def validate(
    model,
    validation_loader,
    valid_name,
    device="mps",
    sw=None,
    epoch_num=-1,
    logger=None,
):
    """Validation on dataset"""
    model.eval()
    val_losses = {"count": 0}
    with torch.no_grad():
        for j, batch in enumerate(validation_loader):
            perf, anchor, label = batch
            anchor = batch[1].float().to(device)
            label = batch[2].long().to(device)

            tot_loss, losses = model.compute_loss(anchor, label)

            if logger and j % 10 == 0:
                logger.info(
                    "step-{} {} {} {} {}".format(
                        j,
                        perf[0],
                        losses["foc_loss"].item(),
                        anchor[0][0][0],
                        label[0],
                    ),
                )

            val_losses["count"] += 1
            for key, value in losses.items():
                if key not in val_losses:
                    val_losses[key] = 0.0
                val_losses[key] += value.item()

        log_str = f"{valid_name}: "
        for key, value in val_losses.items():
            if key == "count":
                continue
            value = value / val_losses["count"]
            log_str = log_str + f"{key}-{value:.3f} "
            if sw is not None:
                sw.add_scalar(f"csi_{valid_name}/{key}", value, epoch_num)
    # if logger:
    #   logger.info(log_str)
    return val_losses


# Unused
# def _calc_label(model, query_loader):
#     query_label = {}
#     query_pred = {}
#     with torch.no_grad():
#         for _j, batch in enumerate(query_loader):
#             perf_b, anchor_b, label_b = batch
#             anchor_b = batch[1].float().to(model.device)
#             label_b = batch[2].long().to(model.device)

#             _, pred_b = model.inference(anchor_b)
#             pred_b = pred_b.cpu().numpy()
#             label_b = label_b.cpu().numpy()

#             for idx_embed in range(len(pred_b)):
#                 perf = perf_b[idx_embed]
#                 pred_embed = pred_b[idx_embed]
#                 pred_label = np.argmax(pred_embed)
#                 prob = pred_embed[pred_label]
#                 label = label_b[idx_embed]
#                 assert np.shape(pred_embed) == (
#                     model.get_ce_embed_length(),
#                 ), f"invalid embed shape:{np.shape(pred_embed)}"
#                 if perf not in query_label:
#                     query_label[perf] = label
#                 else:
#                     assert query_label[perf] == label

#                 if perf not in query_pred:
#                     query_pred[perf] = []
#                 query_pred[perf].append((pred_label, prob))

#     query_perf_label = sorted(query_label.items())
#     return query_perf_label, query_pred


# Unused
# def _syn_pred_label(model, valid_loader, valid_name, sw=None, epoch_num=-1) -> None:
#     model.eval()

#     query_perf_label, query_pred = _calc_label(model, valid_loader)

#     perf_right, perf_total = 0, 0
#     right, total = 0, 0
#     for perf, label in query_perf_label:
#         pred_lst = query_pred[perf]
#         total += len(pred_lst)
#         for pred, _ in pred_lst:
#             right = right + 1 if pred == label else right

#         perf_pred = sorted(pred_lst, key=lambda x: x[1], reverse=False)[0][0]
#         perf_total += 1
#         perf_right = perf_right + 1 if perf_pred == label else perf_right

#     perf_acc = perf_right / perf_total
#     acc = right / total
#     if sw is not None:
#         sw.add_scalar(f"coi_{valid_name}/perf_acc", perf_acc, epoch_num)
#         sw.add_scalar(f"coi_{valid_name}/acc", acc, epoch_num)

#     logging.info(f"{valid_name} perf Acc: {perf_acc:.3f}, Total: {perf_total}")
#     logging.info(f"{valid_name} Acc: {acc:.3f}, Total: {total}")
