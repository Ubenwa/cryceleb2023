"""Utils for working with cryceleb2023 in SpeechBrain.

Author
    * David Budaghyan 2023
    * Mirco Ravanelli 2020
"""


import os
import pickle
import zipfile

import speechbrain as sb
import torch
from huggingface_hub import hf_hub_download


def download_data(dest="data"):

    if os.path.exists(os.path.join(dest, "audio", "train")):
        print(
            f"It appears that data is already downloaded. \nIf you think it should be re-downloaded, remove {dest}/ directory and re-run"
        )
        return

    # download data from Huggingface
    for file_name in ["metadata.csv", "audio.zip", "dev_pairs.csv", "test_pairs.csv", "sample_submission.csv"]:

        hf_hub_download(
            repo_id="Ubenwa/CryCeleb2023",
            filename=file_name,
            local_dir=dest,
            repo_type="dataset",
        )

    with zipfile.ZipFile(os.path.join(dest, "audio.zip"), "r") as zip_ref:
        zip_ref.extractall(dest)

    print("Data downloaded to {dest}/ directory")


class CryBrain(sb.core.Brain):
    """Class for speaker embedding training"."""

    def compute_forward(self, batch, stage):
        """Computation pipeline based on a encoder + speaker classifier.

        Data augmentation and environmental corruption are applied to the input speech.
        """
        batch = batch.to(self.device)
        wavs, lens = batch.sig

        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "augment_pipeline"):

            # Applying the augmentation pipeline
            wavs_aug_tot = []
            wavs_aug_tot.append(wavs)
            for count, augment in enumerate(self.hparams.augment_pipeline):
                # Apply augment
                wavs_aug = augment(wavs, lens)

                # Managing speed change
                if wavs_aug.shape[1] > wavs.shape[1]:
                    wavs_aug = wavs_aug[:, 0 : wavs.shape[1]]
                else:
                    zero_sig = torch.zeros_like(wavs)
                    zero_sig[:, 0 : wavs_aug.shape[1]] = wavs_aug
                    wavs_aug = zero_sig

                if self.hparams.concat_augment:
                    wavs_aug_tot.append(wavs_aug)
                else:
                    wavs = wavs_aug
                    wavs_aug_tot[0] = wavs

            wavs = torch.cat(wavs_aug_tot, dim=0)
            self.n_augment = len(wavs_aug_tot)
            lens = torch.cat([lens] * self.n_augment)

        # Feature extraction and normalization
        feats = self.modules.compute_features(wavs)
        feats = self.modules.mean_var_norm(feats, lens)

        # Embeddings + speaker classifier
        embeddings = self.modules.embedding_model(feats)
        classifier_outputs = self.modules.classifier(embeddings)
        return classifier_outputs, lens

    def compute_objectives(self, compute_forward_return, batch, stage):
        """Computes the loss using speaker-id as label."""
        classifier_outputs, lens = compute_forward_return
        uttid = batch.id
        labels, _ = batch.baby_id_encoded

        # Concatenate labels (due to data augmentations)
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "augment_pipeline"):
            labels = torch.cat([labels] * self.n_augment, dim=0)
            uttid = [f"{u}_{i}" for i in range(self.n_augment) for u in uttid]

        loss = self.hparams.compute_cost(classifier_outputs, labels, lens)

        if stage == sb.Stage.TRAIN and hasattr(self.hparams.lr_scheduler, "on_batch_end"):
            self.hparams.lr_scheduler.on_batch_end(self.optimizer)

        predictions = [str(pred.item()) for pred in classifier_outputs.squeeze().argmax(1)]
        targets = [str(t.item()) for t in labels.squeeze()]

        # append the stats, if val, we also append the log_probs
        if stage == sb.Stage.TRAIN:
            self.classification_stats.append(ids=uttid, predictions=predictions, targets=targets)
        elif stage == sb.Stage.VALID:
            self.classification_stats.append(ids=uttid, predictions=predictions, targets=targets)
        return loss

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of an epoch."""

        self.classification_stats = self.hparams.classification_stats()

    def _write_stats(self, stage, epoch):
        """Wrties stats to f"{self.experiment_dir}/stats/{epoch}/{stage}.txt
        Arguments
        ---------
        epoch: int
            the epoch number
        stage: str
            "train" or "test"
        """
        # do this to extract "train", "val", "test" to put in the file path
        stage = stage.__str__().split(".")[-1].lower()
        output_dir = os.path.join(
            self.hparams.experiment_dir,
            "stats",
            str(epoch),
        )
        # create dir if doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # write classwise stats and confusion stats using speechbrain's classification_stats
        classwise_file_path = os.path.join(output_dir, f"classwise_{stage}.txt")
        with open(classwise_file_path, "w") as w:
            self.classification_stats.write_stats(w)
            # logger.info("classwise_statsvwritten to file: %s", classwise_file_path)

        # write instancewise stats
        instancewise_stats = [["instance_id", "prediction", "target"]]
        instancewise_stats.extend(
            [
                [instance_id, prediction, target]
                for instance_id, prediction, target in zip(
                    self.classification_stats.ids,
                    self.classification_stats.predictions,
                    self.classification_stats.targets,
                )
            ]
        )
        instancewise_file_path = os.path.join(output_dir, f"instancewise_{stage}.pkl")
        with open(instancewise_file_path, "wb") as pkl_file:
            pickle.dump(instancewise_stats, pkl_file)

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch."""

        log_stats = {
            "loss": stage_loss,
            # .summarize("accuracy") computes many statistics
            # but only returns the accuracy. The entire dictionary of stats is written to the stats dir.
            "acc": self.classification_stats.summarize("accuracy") * 100,
        }
        if stage == sb.Stage.TRAIN:
            # this is to save the log_stats, so we
            # write it along the validation log_stats after validation stage
            self.train_log_stats = log_stats
        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            if self.hparams.lrsched_name == 'onplateau':
                old_lr, new_lr = self.hparams.lr_scheduler([self.optimizer], 
                                                           current_epoch=epoch, 
                                                           current_loss=stage_loss)
            else:
                old_lr, new_lr = self.hparams.lr_scheduler(epoch)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
            # LOGGING
            self.hparams.train_logger.log_stats(
                {"Epoch": epoch, "lr": old_lr},
                # train_stats={"loss": self.train_loss}, #do this if only keeping loss during training
                train_stats=self.train_log_stats,
                valid_stats=log_stats,
            )
            # Save the current checkpoint and delete previous checkpoints,
            self.checkpointer.save_and_keep_only(
                name=f"epoch-{epoch}_valacc-{log_stats['acc']:.2f}", meta=log_stats, num_to_keep=4, max_keys=["acc"]
            )

        self._write_stats(stage=stage, epoch=epoch)
