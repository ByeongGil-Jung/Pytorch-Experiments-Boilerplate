from copy import deepcopy
import pickle
import time

import torch

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from trainer.base import TrainerBase

time.time()


class CNNCustomTrainer(TrainerBase):

    def __init__(self, model, model_file_metadata, train_loader, val_loader, test_loader, hyperparameters, tqdm_env='script'):
        super(CNNCustomTrainer, self).__init__(model, model_file_metadata, train_loader, val_loader, test_loader, hyperparameters, tqdm_env)

    # def train(self, optimizer_cls, criterion, n_epoch, lr, hypothesis_threshold, weight_decay):
    def train(self):
        self.model.train()

        # Set hyperparameters
        optimizer_cls = self.hyperparameters.optimizer_cls
        lr = self.hyperparameters.lr
        weight_decay = self.hyperparameters.weight_decay
        n_epoch = self.hyperparameters.n_epoch
        criterion = self.hyperparameters.criterion
        hypothesis_threshold = self.hyperparameters.hypothesis_threshold
        device = self.hyperparameters.device

        optimizer = optimizer_cls(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        best_acc_epoch = 0
        best_f1_epoch = 0
        best_acc = 0
        best_f1 = 0

        train_result_dict_list = list()
        val_result_dict_list = list()

        for epoch in self.tqdm.tqdm(range(n_epoch)):
            total_loss = 0
            n_batch = 0

            label_list = list()
            pred_list = list()

            for i, (img_batch, label_batch) in enumerate(self.train_loader):
                img_batch = img_batch.to(device)
                label_batch = label_batch.to(device).float()

                # Optimization
                optimizer.zero_grad()
                hypothesis = self.model(img_batch)
                loss = criterion(hypothesis, label_batch)
                loss.backward()
                optimizer.step()

                pred = (hypothesis > hypothesis_threshold).float()

                label_list.append(label_batch)
                pred_list.append(pred)
                total_loss += loss.item()
                n_batch += 1

            label_list = torch.cat(label_list).cpu().numpy()
            pred_list = torch.cat(pred_list).cpu().numpy()
            current_loss = total_loss / n_batch

            train_result_dict = dict(
                loss=current_loss,
                accuracy=accuracy_score(y_true=label_list, y_pred=pred_list),
                precision=precision_score(y_true=label_list, y_pred=pred_list),
                recall=recall_score(y_true=label_list, y_pred=pred_list),
                f1=f1_score(y_true=label_list, y_pred=pred_list)
            )
            val_result_dict = self.validate()

            if best_acc < val_result_dict['accuracy']:
                best_acc = val_result_dict['accuracy']
                best_acc_epoch = epoch

            if best_f1 < val_result_dict['f1']:
                best_f1 = val_result_dict['f1']
                best_f1_epoch = epoch
                self.best_model_state_dict = deepcopy(self.model.state_dict())

            print(f"[Epoch - {epoch}]")
            print(f"Train - Accuracy : {round(train_result_dict['accuracy'], 7)}, "
                  f"Precision : {round(train_result_dict['precision'], 7)}, "
                  f"Recall : {round(train_result_dict['recall'], 7)}, "
                  f"F1 : {round(train_result_dict['f1'], 7)}")
            print(f"Val - Accuracy : {round(val_result_dict['accuracy'], 7)}, "
                  f"Precision : {round(val_result_dict['precision'], 7)}, "
                  f"Recall : {round(val_result_dict['recall'], 7)}, "
                  f"F1 : {round(val_result_dict['f1'], 7)}")
            print(f"Best Accuracy : {best_acc} (epoch : {best_acc_epoch})")
            print(f"Best F1 : {best_f1} (epoch : {best_f1_epoch})")

            # Save Model & Record dict
            record_dict = dict(
                train_result_dict=train_result_dict,
                val_result_dict=val_result_dict
            )

            torch.save(self.model.state_dict(), self.model_file_metadata.get_save_model_file_path(epoch=epoch))

            with open(self.model_file_metadata.get_save_record_file_path(epoch=epoch), "wb") as f:
                pickle.dump(record_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

            train_result_dict_list.append(train_result_dict)
            val_result_dict_list.append(val_result_dict)

        # Save last result
        entire_record_dict = dict(
            train_result_dict_list=train_result_dict_list,
            val_result_dict_list=val_result_dict_list
        )
        torch.save(self.best_model_state_dict, self.model_file_metadata.get_best_model_file_path())

        # Save entire_record_dict
        with open(self.model_file_metadata.get_entire_record_file_path(), "wb") as f:
            pickle.dump(entire_record_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Load best model
        self.load_best_model()

        return entire_record_dict

    def validate(self):
        self.model.eval()

        # Set hyperparameters
        criterion = self.hyperparameters.criterion
        hypothesis_threshold = self.hyperparameters.hypothesis_threshold
        device = self.hyperparameters.device

        total_loss = 0
        n_batch = 0

        label_list = list()
        pred_list = list()

        for i, (img_batch, label_batch) in enumerate(self.val_loader):
            img_batch = img_batch.to(device)
            label_batch = label_batch.to(device).float()

            with torch.no_grad():
                hypothesis = self.model(img_batch)
                loss = criterion(hypothesis, label_batch)

            pred = (hypothesis > hypothesis_threshold).float()

            label_list.append(label_batch)
            pred_list.append(pred)
            total_loss += loss.item()
            n_batch += 1

        label_list = torch.cat(label_list).cpu().numpy()
        pred_list = torch.cat(pred_list).cpu().numpy()
        current_loss = total_loss / n_batch

        return dict(
            loss=current_loss,
            accuracy=accuracy_score(y_true=label_list, y_pred=pred_list),
            precision=precision_score(y_true=label_list, y_pred=pred_list),
            recall=recall_score(y_true=label_list, y_pred=pred_list),
            f1=f1_score(y_true=label_list, y_pred=pred_list)
        )

    def predict(self):
        return self.validate()
