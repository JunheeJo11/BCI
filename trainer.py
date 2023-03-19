from tqdm import tqdm
from einops import repeat
from einops.layers.torch import Rearrange

import torch
import torch.nn as nn
import torch.optim as optim

from model import Model
import warnings
import wandb

warnings.filterwarnings('ignore')


class Trainer:
    def __init__(self, args):
        self.args = args
        self.model = Model(self.args).to(self.args.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(self.args.device)
        # self.criterion = nn.BCEWithLogitsLoss().to(self.args.device)

        self.best_acc = 0

    def train(self, train_iter, val_iter):
        if self.args.train_mode == 'llt':
            self.llt_train(train_iter, val_iter)

        if self.args.train_mode == 'hlt':
            self.hlt_train(train_iter, val_iter)
            return self.best_acc

    def llt_train(self, train_iter, val_iter):

        # with wandb.init(project="BCI"):
        #     wandb.run.name = 'subject 1.fixed.llt.BCEWithLogitsLoss'
        for epoch in range(self.args.epochs):
            batch_loss, batch_acc = 0, 0

            self.model.train()
            for source, target in tqdm(train_iter):
                # y=[]
                source = source.to(self.args.device)
                target1 = repeat(target, 'h c -> (h c r)', r=self.args.seq_len).to(self.args.device)
                # target1 = repeat(target, 'h c -> (h r) c', r=self.args.seq_len).to(self.args.device)

                self.optimizer.zero_grad()
                y1, _, _ = self.model(source)  # 맨뒤에 _생략
                # target1 = repeat(target, 'h c -> (h c r)', r=self.args.seq_len_hlt).type(torch.LongTensor).to(self.args.device)
                # y1 = torch.argmax(y1, dim=1)
                # for i in range(len(y1)):
                #     y.append(y1[i][torch.argmax(y1, dim=1)[i]].item())
                # y = torch.tensor(y).to(self.args.device)
                loss = self.criterion(y1, target1.float())
                # loss = self.criterion(y.float(), target1.float())
                # loss.requires_grad_(True)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                self.optimizer.step()

                batch_loss += loss.item()
                batch_acc += self.metric(target1, y1)

            train_loss = batch_loss / len(train_iter)


            val_loss1, val_loss2, val_acc1, val_acc2 = self.evaluate(val_iter)
            val_acc1 = val_acc1 / (self.args.val_len * self.args.seq_len)

            if self.best_acc < val_acc1:
                self.best_acc = val_acc1
                self.save_param(param=self.model, filename='LLT.BCEWithLogitsLoss')

            self.save_param(param=self.optimizer, filename='OPTIM.BCEWithLogitsLoss')

            print(f'Subject: {self.args.eval_subject}, Epoch:  {epoch}, Length: {self.args.seq_len_hlt}')
            print(
                f'Train Loss Lv 1: {train_loss:.3f} \t Val. Loss Lv 1: {val_loss1:.3f} \t Val. Acc Lv 1: {val_acc1: .3f}')
            print('\n ')
                # wandb.log({'train_loss': train_loss, 'val_loss': val_loss1, 'accuracy': val_acc1}, step=epoch)

    def hlt_train(self, train_iter, val_iter):
        self.args.how_train = 'fixed'
        param = self.load_param(filename='LLT.BCEWithLogitsLoss')
        self.model.load_state_dict(param['model'])
        self.optimizer.load_state_dict(param['optim'])
        self.model.llt_freeze()
        self.best_acc = 0

        # with wandb.init(project="BCI"):
        #     wandb.run.name = 'subject 1.variable.hlt.BCEWithLogitsLoss'
        for epoch in range(self.args.epochs):
            batch_loss, batch_acc = 0, 0

            self.model.train()
            for source, target in tqdm(train_iter):
                source = source.to(self.args.device)
                # target = Rearrange('b c -> (b c)', c=1)(target).to(self.args.device)
                # target = target.type(torch.LongTensor).to(self.args.device)


                self.optimizer.zero_grad()
                _, y, _ = self.model(source)

                # target = Rearrange('b c -> (b c)', c=1)(target).type(torch.LongTensor).to(self.args.device)
                target = target.type(torch.LongTensor).to(self.args.device)
                loss = self.criterion(y, target.float())
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                self.optimizer.step()

                batch_loss += loss.item()
                batch_acc += self.metric(target, y)

            train_loss = batch_loss / len(train_iter)

            val_loss1, val_loss2, val_acc1, val_acc2 = self.evaluate(val_iter)
            val_acc2 = val_acc2 / self.args.val_len

            if self.best_acc < val_acc2:
                self.best_acc = val_acc2
                # self.save_param(param=self.model, filename=f'HLT_{self.args.seq_len_hlt}')
                self.args.how_train = 'variable'
                self.save_param(param=self.model, filename=f'HLT.BCEWithLogitsLoss')
            print('\n ')
            print(f'Subject: {self.args.eval_subject}, Epoch:  {epoch}, Length: {self.args.seq_len_hlt}')
            print(
                f'Train Loss Lv 2: {train_loss:.3f} \t Val. Loss Lv 2: {val_loss2:.3f} \t Val. Acc Lv 2: {val_acc2: .3f}')
            print('\n ')
                # wandb.log({'train_loss': train_loss, 'val_loss': val_loss2, 'accuracy': val_acc2}, step=epoch)

    @torch.no_grad()
    def evaluate(self, val_iter):
        batch_loss1, valid_acc1, batch_loss2, valid_acc2 = 0, 0, 0, 0
        self.model.eval()
        for source, target in tqdm(val_iter):
            source = source.to(self.args.device)
            y_1 = []
            y_2 = []

            y1, y2, _ = self.model(source)  # 중간에 y2를 _로 변경

            for i in range(len(y1)):
                y_1.append(y1[i][torch.argmax(y1, dim=1)[i]].item())
            for i in range(len(y2)):
                y_2.append(y2[i][torch.argmax(y2, dim=1)[i]].item())
            y_1 = torch.tensor(y_1).to(self.args.device)
            y_2 = torch.tensor(y_2).to(self.args.device)

            # target1 = repeat(target, 'h c -> (h r) c', r=self.args.seq_len_hlt).type(torch.LongTensor).to(self.args.device)
            # target2 = Rearrange('b c -> (b c)', c=1)(target).type(torch.LongTensor).to(self.args.device)
            # target1 = repeat(target, 'h c -> (h r) c', r=self.args.seq_len_hlt).to(self.args.device)
            # target2 = target.type(torch.LongTensor).to(self.args.device)
            target1 = repeat(target, 'h c -> (h r c)', r=self.args.seq_len_hlt).to(self.args.device)
            target2 = target.squeeze().type(torch.LongTensor).to(self.args.device)


            loss1 = self.criterion(y_1.float(), target1.float())
            loss2 = self.criterion(y_2.float(), target2.float())
            loss1.requires_grad_(True)
            loss2.requires_grad_(True)

            batch_loss1 += loss1.item()
            valid_acc1 += self.metric(target1, y1)

            batch_loss2 += loss2.item()
            valid_acc2 += self.metric(target2, y2)

        valid_loss1 = batch_loss1 / len(val_iter)
        valid_loss2 = batch_loss2 / len(val_iter)
        return valid_loss1, valid_loss2, valid_acc1, valid_acc2

    def metric(self, target, output):
        acc = 0
        out = torch.argmax(output, dim=1)
        for i in range(target.shape[0]):
            if target[i] == out[i]:
                acc += 1
        return acc

    def save_param(self, param, filename):
        # param = {'param': param.state_dict(),
        #          'best_acc': self.best_acc}
        param = {'model': param.state_dict(),
                 'optim': self.optimizer.state_dict(),
                 'best_acc': self.best_acc}
        # torch.save(param, f'weight/{self.args.dataset}_{filename}_S{self.args.eval_subject}_{self.args.eval_idx}.pth')
        torch.save(param, f'weight/{self.args.dataset}_{filename}_{self.args.how_train}_S{self.args.eval_subject}_{self.args.eval_idx}.pth')


    def load_param(self, filename):
        # param = torch.load(f'weight/{self.args.dataset}_{filename}_S{self.args.eval_subject}_{self.args.eval_idx}.pth')
        param = torch.load(f'weight/{self.args.dataset}_{filename}_{self.args.how_train}_S{self.args.eval_subject}_{self.args.eval_idx}.pth')
        return param

    def inference(self, x):
        # param = self.load_param(filename=f'HLT_{self.args.seq_len_hlt}')
        param = self.load_param(filename=f'HLT')
        self.model.load_state_dict(param['model'])
        self.model.eval()
        x = torch.tensor(x, dtype=torch.float32)
        x = x.to(self.args.device)
        _, y_hat2, score = self.model(x)

        y_hat2 = torch.nn.Softmax()(y_hat2)  # score 삭제

        # wandb.log({"y_hat2": y_hat2})

        # print(y_hat2)
        # y_hat2 = torch.argmax(y_hat2, dim=1)
        # print(y_hat2)
        return y_hat2, score  # score 삭제
