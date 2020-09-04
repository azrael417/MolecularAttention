import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics


class ComplexPytorchHistory:
    def __init__(self, metric=metrics.r2_score, metric_name='r2', classifacation=False):
        self.train_loss = []
        self.test_loss = []
        self.train_r2 = []
        self.test_r2 = []
        self.max_r2_train = []
        self.min_r2_train = []
        self.std_r2_train = []
        self.median_r2_train = []

        self.max_r2_test = []
        self.min_r2_test = []
        self.std_r2_test = []
        self.median_r2_test = []
        self.metric = metric
        self.metric_name = metric_name
        self.true_tracker, self.pred_tracker = [], []
        self.init = True

    def log_loss(self, loss, train=True):
        if train:
            self.train_loss.append(loss)
        else:
            self.test_loss.append(loss)

    def track_metric(self, pred, value, mask=None):
        vs = pred.shape[-1]
        if mask is not None:
            mask = mask.astype(np.bool)
        if self.init:
            for i in range(vs):
                self.true_tracker.append([])
                self.pred_tracker.append([])
            self.init = False

        for i in range(vs):
            if mask is not None:
                self.pred_tracker[i].append(pred[np.where(mask[:, i]), i].flatten())
                self.true_tracker[i].append(value[np.where(mask[:, i]), i].flatten())
            else:
                self.pred_tracker[i].append(pred[:, i])
                self.true_tracker[i].append(value[:, i])

    def get_last_metric(self, train=True):
        if train:
            return self.train_r2[-1], self.std_r2_train[-1], self.median_r2_train[-1], self.max_r2_train[-1], \
                   self.min_r2_train[-1]
        else:
            return self.test_r2[-1], self.std_r2_test[-1], self.median_r2_test[-1], self.max_r2_test[-1], \
                   self.min_r2_test[-1]

    def log_metric(self, r2=None, train=True, internal=False, avg_met=None):
        if internal:
            avg_met = []
            for i in range(len(self.true_tracker)):
                true_tracker = np.concatenate(self.true_tracker[i]).flatten()
                pred_tracker = np.concatenate(self.pred_tracker[i]).flatten()
                true_tracker = np.nan_to_num(true_tracker, nan=0, posinf=0, neginf=0)
                pred_tracker = np.nan_to_num(pred_tracker, nan=0, posinf=0, neginf=0)
                r2 = self.metric(true_tracker, pred_tracker)
                avg_met.append(max(0, r2))
            self.true_tracker, self.pred_tracker = [], []
            self.init = True
            r2 = np.mean(avg_met)
        if train:
            self.train_r2.append(r2)
            self.std_r2_train.append(np.std(avg_met))
            self.median_r2_train.append(np.median(avg_met))
            self.max_r2_train.append(np.percentile(avg_met, 5))
            self.min_r2_train.append(np.percentile(avg_met, 95))
        else:
            self.test_r2.append(r2)
            self.std_r2_test.append(np.std(avg_met))
            self.median_r2_test.append(np.median(avg_met))
            self.max_r2_test.append(np.percentile(avg_met, 5))
            self.min_r2_test.append(np.percentile(avg_met, 95))

    def plot_loss(self, save_file=None, title='Loss', figsize=(8, 5)):
        pass

    def plot_metric(self, save_file=None, title='Loss', figsize=(8, 5)):
        pass


class PytorchHistory:
    def __init__(self, metric=metrics.r2_score, metric_name='r2'):
        self.train_loss = []
        self.test_loss = []
        self.train_r2 = []
        self.test_r2 = []
        self.metric = metric
        self.metric_name = metric_name
        self.true_tracker, self.pred_tracker = [], []

    def log_loss(self, loss, train=True):
        if train:
            self.train_loss.append(loss)
        else:
            self.test_loss.append(loss)

    def track_metric(self, pred, value):
        self.pred_tracker.append(pred)
        self.true_tracker.append(value)

    def get_last_metric(self, train=True):
        if train:
            return self.train_r2[-1]
        else:
            return self.test_r2[-1]

    def log_metric(self, r2=None, train=True, internal=False):
        if internal:
            self.true_tracker = np.concatenate(self.true_tracker).flatten()
            self.pred_tracker = np.concatenate(self.pred_tracker).flatten()
            self.true_tracker = np.nan_to_num(self.true_tracker, nan=0, posinf=0, neginf=0)
            self.pred_tracker = np.nan_to_num(self.pred_tracker, nan=0, posinf=0, neginf=0)
            r2 = self.metric(self.true_tracker, self.pred_tracker)
            # print("train" if train else "test", sklearn.metrics.confusion_matrix(self.true_tracker, self.pred_tracker))
            self.true_tracker, self.pred_tracker = [], []
        if train:
            self.train_r2.append(r2)
        else:
            self.test_r2.append(r2)

    def plot_loss(self, save_file=None, title='Loss', figsize=(8, 5)):
        plt.figure(figsize=figsize)

        plt.plot(list(range(len(self.train_loss))), self.train_loss, label='Train Loss')
        plt.plot(list(range(len(self.test_loss))), self.test_loss, label='Test Loss')

        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")

        if save_file is None:
            plt.show()
        else:
            plt.savefig(save_file, bbox_inches='tight', dpi=300)

    def plot_metric(self, save_file=None, title='Loss', figsize=(8, 5)):
        plt.figure(figsize=figsize)

        plt.plot(list(range(len(self.train_r2))), self.train_r2, label='Train r2')
        plt.plot(list(range(len(self.test_r2))), self.test_r2, label='Test r2')

        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel(self.metric_name)

        if save_file is None:
            plt.show()
        else:
            plt.savefig(save_file, bbox_inches='tight', dpi=300)
