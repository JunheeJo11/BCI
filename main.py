import torchvision.transforms

from utils import TenFCVDataset, LOSOCVDataset, ArgCenter
from utils import MyDataset
from torch.utils.data import DataLoader
from trainer import Trainer
import numpy as np
import matplotlib.pyplot as plt
import torch
import random
import pickle
from PIL import Image
import wandb

class Train:
    def __init__(self, dataset):
        self.args = ArgCenter(dataset).get_arg()

    def custom_collate(self, batch):
        length = 3*random.randint(10, 333)
        self.args.seq_len_hlt = length//3 -1
        _data = torch.empty(size=(len(batch), length, 62))
        _label = torch.empty(size=(len(batch), 1))
        for i in range(len(batch)):
            _data[i] = batch[i][0][:length]
            _label[i] = batch[i][1]

        return _data, _label

    def SVtrain(self, subject, fold, how):
        self.args.eval_idx = fold
        self.args.eval_subject = subject
        self.args.seq_len_hlt = 64
        # self.args.epochs = 10

        tfcv = TenFCVDataset(subject=subject, args=self.args, fold=self.args.eval_idx)
        x_train, y_train, x_val, y_val = tfcv.get_data()

        self.args.val_len = x_val.shape[0]

        train_loader = MyDataset(x_train, y_train)
        val_loader = MyDataset(x_val, y_val)

        # train_iter = DataLoader(dataset=train_loader, batch_size=self.args.batch_size, shuffle=True,
        #                         num_workers=self.args.num_workers)
        # val_iter = DataLoader(dataset=val_loader, batch_size=self.args.batch_size * 4, shuffle=False,
        #                       num_workers=self.args.num_workers)
        if how == "fixed":
            train_iter = DataLoader(dataset=train_loader, batch_size=self.args.batch_size, shuffle=True,
                                    num_workers=self.args.num_workers)
            val_iter = DataLoader(dataset=val_loader, batch_size=self.args.batch_size*4, shuffle=False,
                                  num_workers=self.args.num_workers)
            self.args.how_train = "fixed"

            trainer = Trainer(self.args)

            self.args.train_mode = 'llt'
            trainer.train(train_iter, val_iter)

            self.args.train_mode = 'hlt'
            trainer.train(train_iter, val_iter)

        elif how == "variable":
            train_iter = DataLoader(dataset=train_loader, batch_size=self.args.batch_size, shuffle=True,
                                    num_workers=self.args.num_workers, collate_fn=self.custom_collate)
            val_iter = DataLoader(dataset=val_loader, batch_size=self.args.batch_size * 4, shuffle=False,
                                  num_workers=self.args.num_workers, collate_fn=self.custom_collate)
            self.args.how_train = "variable"

            trainer = Trainer(self.args)

            self.args.train_mode = 'hlt'
            trainer.train(train_iter, val_iter)

        # trainer = Trainer(self.args)
        #
        # self.args.train_mode = 'llt'
        # trainer.train(train_iter, val_iter)
        #
        # self.args.train_mode = 'hlt'
        # trainer.train(train_iter, val_iter)

    def SItrain(self, subject):
        lsv = LOSOCVDataset(args=self.args, eval_subj_index=subject)
        x_train, y_train = lsv._train_data, lsv._train_label
        x_val, y_val = lsv.get_eval_data_label()

        self.args.val_len = x_val.shape[0]
        self.args.eval_subject = subject

        train_loader = MyDataset(x_train, y_train)
        val_loader = MyDataset(x_val, y_val)

        train_iter = DataLoader(dataset=train_loader, batch_size=self.args.batch_size, shuffle=True,
                                num_workers=self.args.num_workers)
        val_iter = DataLoader(dataset=val_loader, batch_size=self.args.batch_size * 4, shuffle=False,
                              num_workers=self.args.num_workers)

        trainer = Trainer(self.args)

        self.args.train_mode = 'llt'
        trainer.train(train_iter, val_iter)

        self.args.train_mode = 'hlt'
        acc = trainer.train(train_iter, val_iter)

        report = f'{self.args.dataset},{self.args.eval_subject},{acc}\n'
        writes = open('sample.txt','a')
        writes.write(report)


    def subject_category_evaluation(self, subject):

        lsv = LOSOCVDataset(args=self.args, eval_subj_index=subject)
        x_val, y_val = lsv.get_eval_data_label()

        x_val = x_val[:, :, :]
        # x_val = x_val
        y_val = y_val.squeeze()
        # print(x_val.shape)
        # print(x_val)
        # print(y_val.shape)

        self.args.val_len = x_val.shape[0]
        self.args.eval_subject = subject

        # for j in range(1,66):
        self.args.num_token = 65

        trainer = Trainer(self.args)
        y_hat, score = trainer.inference(x_val)
        y_hat = y_hat.cpu().detach().numpy()
        # y_hat = np.asarray(y_hat.cpu())

        # x_val = np.max(x_val, axis=1)
        # x_val = np.max(x_val, axis=0)
        # x_val = np.mean(x_val, axis=2)
        # print(x_val.shape)
        # print(x_val)

        TP = 0
        for i in range(self.args.val_len):
            if (y_hat[i] == y_val[i]).all():
                    TP +=1
        acc = TP/self.args.val_len
        # print(f'{j}: {acc}')

        print(y_hat)
        # print(acc)
        # print(y_hat.shape, type(y_hat))
        # right_res = []
        # for i in range(len(y_hat)):
        #     right_res.append(y_hat[i][1])
        #
        # left_res = []
        # for i in range(len(y_hat)):
        #     left_res.append(y_hat[i][0])
        #
        # data = x_val[0,:,0]
        # for i in range(len(x_val)):
        #     data.append(x_val[0,:,0])

        return acc
        pass

    def subject_dependent_category_evaluation(self, subject, fold):

        tfcv = TenFCVDataset(args=self.args, subject=subject, fold=fold)
        x_val, y_val, for_test = tfcv.get_eval_data()

        print(for_test)
        result = np.zeros((40,66))
        accs = np.zeros((40, 66))
        y_hats = []
        step = 0
        time_accuracy = np.zeros(66)
        # threshold_accuracy = np.zeros(50)

        threshold = 0
        prob = 0
        count = 0
        threshold_accuracy = 0
        mean_time = 0
        times = np.zeros(40)
        probs = np.zeros(40)
        mean_times = np.zeros(50)
        threshold_accuracys = np.zeros(50)
        for l in range(50):
            print("{}th threshold".format(l+1))
            print("threshold is {}".format(1-threshold))
            for k in for_test:
                idx = np.where(for_test == k)
                # print(idx[0][0])
                s_val = x_val[k-1:k, :, :]
                print(s_val.shape)
                print("answer is {}".format(y_val[k-1]))
                print("{}th trial".format(idx[0][0]+1))

                self.args.val_len = x_val.shape[0]
                self.args.eval_subject = subject
                for j in range(1, 67):
                    # self.args.num_token = j
                    self.args.seq_len_hlt = j

                    time = 0
                    trainer = Trainer(self.args)
                    y_hat, score = trainer.inference(s_val)
                    y_hat = y_hat.cpu().detach().numpy()
                    print(y_hat)
                    # print("{}th token's answer is {} and max is {}".format(j, y_hat, np.max(y_hat)))
                    # 왼손이나 오른손 확률이 threshold 넘어가면 시간과 0/1 기록
                    if np.max(y_hat) >= (1-threshold):
                        print("early classified")
                        if time == 0:
                            time = j
                        elif time < j:
                            time = time
                        if y_hat[0][0] > y_hat[0][1]:
                            prob = 0
                        else:
                            prob = 1
                    elif j == 66:
                        time = 66
                        if y_hat[0][0] > y_hat[0][1]:
                            prob = 0
                        else:
                            prob = 1
                    times[idx[0][0]] = time
                    # print("time is", times)
                    probs[idx[0][0]] = prob

                for m in range(40):
                    if y_val[k-1] == probs[m]:
                        count += 1
            print(times)
            print(probs)
            # print(count)
            mean_time = np.average(times)
            threshold_accuracy = count/40
            print("mean_time of {}th threshold is {}".format(l, mean_time))
            print("threshold_accuracy of {}th threshold is {}".format(l, threshold_accuracy))
            mean_times[l] = mean_time
            threshold_accuracys[l] = threshold_accuracy
            count = 0
            threshold += 0.01
            times = np.zeros(40)

        plt.subplot(2,1,1)
        plt.plot(mean_times, threshold_accuracys)
        plt.show()

        for k in for_test:
            idx = np.where(for_test == k)
            s_val = x_val[k-1:k, :, :]
            print("{}th trial".format(idx[0][0]))
            # print(s_val.shape)

            # self.args.val_len = s_val.shape(0)
            self.args.val_len = 1
            self.args.eval_subject = subject

            # x_val = x_val[397:398, :, :]
            # y_val = y_val[173]
            # print(y_val)
            # x_val = x_val
            # y_val = y_val.squeeze()
            # print(x_val.shape)
            # print(x_val)
            # print(y_val.shape)
            # print(y_val)

            # self.args.val_len = x_val.shape[0]
            # self.args.eval_subject = subject

            for j in range(1,67):
                # self.args.num_token = j
                self.args.seq_len_hlt = j

                trainer = Trainer(self.args)
                y_hat, score = trainer.inference(s_val)
                y_hat = y_hat.cpu().detach().numpy()
                # print(y_hat)

                # accuracy by the time
                max_idx = np.argmax(y_hat)
                result[step][j - 1] = max_idx
                # print(max_idx)
                if max_idx == y_val[k-1]:
                    accs[step][j-1] = 1
                else:
                    accs[step][j-1] = 0
                # print(result.shape)
            step += 1

            # y_hat = np.asarray(y_hat.cpu())

            # x_val = np.max(x_val, axis=1)
            # x_val = np.max(x_val, axis=0)
            # x_val = np.mean(x_val, axis=2)
            # print(x_val.shape)
            # print(x_val)

                # TP = 0
                # for i in range(self.args.val_len):
                #     if (y_hat[i] == y_val[i]).all():
                #             TP +=1
                # acc = TP/self.args.val_len
                # print(f'{j}: {acc}')

                # print(y_hat.shape)
                # print("{} token: ".format(j), y_hat)
                # y_hats.append(y_hat[0][1])

        # accuracy by the time
        for i in range(66):
            sum = 0
            for j in range(40):
                sum += accs[j][i]
            time_accuracy[i] = sum / 40
        # print(time_accuracy)

        plt.subplot(2,1,2)
        plt.plot(time_accuracy)
        plt.show()

        # print(result)
        # print(accs)


        # plt.plot(time_accuracy)
        # plt.show()

        # n=1
        # while n<41:
        #    for k in for_test:
        #
        #        if y_val[k-1] == 0:
        #            print("label is 0, n is {}".format(n))
        #            plt.subplot(1, 2, 1)
        #            plt.plot(y_hats[66 * n - 66:66 * n])
        #            n+=1
        #        else:
        #            print("label is 1, n is {}".format(n))
        #            plt.subplot(1, 2, 2)
        #            plt.plot(y_hats[66 * n - 66:66 * n])
        #            n+=1
        # plt.show()

        # n = 1
        # while n < 41:
        #     print("label is 0, n is {}".format(n))
        #     plt.subplot(1, 2, 1)
        #     plt.plot(y_hats[66 * n - 66:66 * n])
        #     n += 1
        # plt.show()


        # for n in range(1,11):
        #     if y_val[k-1] == 0:
        #         plt.subplot(1, 2, 1)
        #         plt.plot(y_hats[66 * n - 66:66 * n])
        #     elif y_val[k-1] == 1:
        #         plt.subplot(1, 2, 2)
        #         plt.plot(y_hats[66 * n - 66:66 * n])
        # plt.show()

            # print(acc)
        # print(acc)
        # print(y_hat.shape, type(y_hat))
        # right_res = []
        # for i in range(len(y_hat)):
        #     right_res.append(y_hat[i][1])
        #
        # left_res = []
        # for i in range(len(y_hat)):
        #     left_res.append(y_hat[i][0])
        #
        # data = x_val[0,:,0]
        # for i in range(len(x_val)):
        #     data.append(x_val[0,:,0])


        return  # x_val, y_hat, for_test
        pass

    def dep_eval(self, subject, fold):   # full length input으로 학습시킨거 그래프 그릴 때 사용

        self.args.eval_idx = fold
        self.args.eval_subject = subject
        self.args.seq_len_hlt = 64

        tfcv = TenFCVDataset(args=self.args, subject=self.args.eval_subject, fold=self.args.eval_idx)
        x_axis = [n for n in range(1, 65)]
        answers = []
        left = []
        right = []
        fixed = np.zeros((40,64))
        fixed_acc = []
        prediction = np.zeros((40, 64))
        variable = np.zeros((50, 40))
        variable_acc = []
        times = np.zeros((50, 40))
        mean_time = []


        for j in range(0, 40):
            print(f"{j} trial")
            count = 0
            for i in range(10, 334):
                count += 1
                x_train, y_train, x_val, y_val = tfcv.get_data()
                self.args.seq_len_hlt = 64
                length = 3*i
                x_val = x_val[j:j+1, :length, :]
                y_val = y_val[j:j+1, :]

                self.args.val_len = x_val.shape[0]

                trainer = Trainer(self.args)

                yhat, score = trainer.inference(x_val)
                answer = int(y_val[0])

                if count % 5 == 0:
                    prediction[j][(count//5)-1] = yhat[0][1].item()
                    if int(y_val[0]) == 0:
                        answer = 0
                        left.append(yhat[0][1].item())
                    else:
                        answer = 1
                        right.append(yhat[0][1].item())
                    print(f"length is {self.args.seq_len_hlt}", f'prediction is {yhat}', f'the answer is {y_val[0]}')

            answers.append(answer)
        print("answers is ", np.shape(answers), answers)
        print("left is ", np.shape(left), left)
        print("right is ", np.shape(right), right)
        print("prediction is ", np.shape(prediction), prediction)

        with open(f"Lee_S{subject}_full_answers.pkl", "wb") as f:
            pickle.dump(answers, f)
        with open(f"Lee_S{subject}_full_left.pkl", "wb") as f:
            pickle.dump(left, f)
        with open(f"Lee_S{subject}_full_right.pkl", "wb") as f:
            pickle.dump(right, f)
        with open(f"Lee_S{subject}_full_prediction.pkl", "wb") as f:
            pickle.dump(prediction, f)

        # with open(f"Lee_S{subject}_full_answers.pkl", "rb") as f:
        #     answers = pickle.load(f)
        # with open(f"Lee_S{subject}_full_left.pkl", "rb") as f:
        #     left = pickle.load(f)
        # with open(f"Lee_S{subject}_full_right.pkl", "rb") as f:
        #     right = pickle.load(f)
        # with open(f"Lee_S{subject}_full_prediction.pkl", "rb") as f:
        #     prediction = pickle.load(f)

        print("left is ", np.shape(left), left)
        print("right is ", np.shape(right), right)
        print(len(left), len(right))

        #drawing left/right hand prob graph
        for i in range(int(len(left) / 64)):
            plt.plot(x_axis, left[i*64:(i+1)*64])
            plt.title(f"Lee_S{subject}_LH_{i+1}")
            plt.xlabel("time(token)")
            plt.ylabel("Prob")
            plt.ylim([0,1])
        plt.show()
        plt.savefig(f"Lee_S{subject}_LH.png")
        for i in range(int(len(right) / 64)):
            plt.plot(x_axis, right[i*64:(i+1)*64])
            plt.title(f"Lee_S{subject}_RH_{i+1}")
            plt.xlabel("time(token)")
            plt.ylabel("Prob")
            plt.ylim([0,1])
        plt.show()
        plt.savefig(f"Lee_S{subject}_RH.png")

        return

    def inf(self, subject, fold, how):      # 랜덤한 길이의 인풋으로 학습시킨거 그래프 그릴 때 사용

        if how=="fixed":
            self.args.how_train = "fixed"
        elif how=="variable":
            self.args.how_train = "variable"

        self.args.eval_idx = fold
        self.args.eval_subject = subject
        self.args.seq_len_hlt = 64

        tfcv = TenFCVDataset(subject=self.args.eval_subject, args=self.args, fold=self.args.eval_idx)
        x_axis = [n for n in range(1, 65)]
        answers = []
        left = []
        right = []
        fixed = np.zeros((40,64))
        fixed_acc = []
        prediction = np.zeros((40, 64))
        variable = np.zeros((50, 40))
        variable_acc = []
        times = np.zeros((50, 40))
        mean_time = []
        #
        # for j in range(0,40):
        #     print(f"{j} trial")
        #     count = 0
        #     l = []
        #     r = []
        #     answer = 0
        #     for i in range(10,334):
        #         count+=1
        #         x_train, y_train, x_val, y_val = tfcv.get_data()
        #         self.args.seq_len_hlt = 64
        #         length = 3*i
        #         x_val = x_val[j:j+1, :length, :]
        #         y_val = y_val[j:j+1, :]
        #
        #         self.args.val_len = x_val.shape[0]
        #
        #         trainer = Trainer(self.args)
        #
        #         yhat, score = trainer.inference(x_val)
        #         answer = int(y_val[0])
        #         if count % 5 == 0:
        #             r.append(yhat[0][1].item())
        #             prediction[j][(count//5)-1] = yhat[0][1].item()
        #             if int(y_val[0]) == 0:
        #                 answer = 0
        #                 left.append(yhat[0][1].item())
        #             else:
        #                 answer = 1
        #                 right.append(yhat[0][1].item())
        #             print(f"length is {self.args.seq_len_hlt}", f'prediction is {yhat}', f'the answer is {y_val[0]}')
        #
        #     print("r is ", np.shape(r), r)
        #     answers.append(answer)
        # print("answers is ", np.shape(answers), answers)
        # print("left is ", np.shape(left), left)
        # print("right is ", np.shape(right), right)
        # print("prediction is ", np.shape(prediction), prediction)
        #
        # with open(f"result/Lee_S{subject}_{how}_answers.pkl", "wb") as f:
        #     pickle.dump(answers, f)
        # with open(f"result/Lee_S{subject}_{how}_left.pkl", "wb") as f:
        #     pickle.dump(left, f)
        # with open(f"result/Lee_S{subject}_{how}_right.pkl", "wb") as f:
        #     pickle.dump(right, f)
        # with open(f"result/Lee_S{subject}_{how}_prediction.pkl", "wb") as f:
        #     pickle.dump(prediction, f)

        with open(f"result/Lee_S{subject}_{how}_answers.pkl", "rb") as f:
            answers = pickle.load(f)
        with open(f"result/Lee_S{subject}_{how}_left.pkl", "rb") as f:
            left = pickle.load(f)
        with open(f"result/Lee_S{subject}_{how}_right.pkl", "rb") as f:
            right = pickle.load(f)
        with open(f"result/Lee_S{subject}_{how}_prediction.pkl", "rb") as f:
            prediction = pickle.load(f)
        #
        # print("left is ", np.shape(left), left)
        # print("right is ", np.shape(right), right)
        # print(len(left), len(right))


        # drawing left/right hand probability graph
        plt.figure(figsize=(7,5))
        # with wandb.init(project="BCI"):
        #     wandb.run.name = 'probability'
        for i in range(int(len(left) / 64)):
            # data = [[x, y] for (x,y) in zip(x_axis, left[i*64:(i+1)*64])]
            # table = wandb.Table(data=data, columns=["time(token)", "LH_prob"])
            # wandb.log({"left_hand_plot":wandb.plot.line(table, "time(token)", "LH_prob", title="LH_prob")})
            # plt.subplot(1,2,1)
            plt.plot(x_axis, left[i*64:(i+1)*64])
            plt.title(f"Lee_S{subject}_LH_{i+1}")
            plt.xlabel('time(token)')
            plt.ylabel('prob')
            plt.ylim([0,1])
        # plt.show()
        plt.savefig(f"graph/Lee_S{subject}_{how}_LH.png")

        plt.figure(figsize=(7,5))
        for i in range(int(len(right)/64)):
            # plt.subplot(1,2,2)
            plt.plot(x_axis, right[i*64:(i+1)*64])
            plt.title(f"Lee_S{subject}_RH_{i+1}")
            plt.xlabel('time(token)')
            plt.ylabel('prob')
            plt.ylim([0, 1])
        # plt.show()
        plt.savefig(f"graph/Lee_S{subject}_{how}_RH.png")


        # finding fixed accuracy
        for i in range(0,64):
            for j in range(0,40):
                if prediction[j][i] > 0.5:
                    if answers[j] == 1:
                        fixed[j][i] = 1
                    elif answers[j] == 0:
                        fixed[j][i] = 0
                elif prediction[j][i] < 0.5:
                    if answers[j] == 1:
                        fixed[j][i] = 0
                    elif answers[j] == 0:
                        fixed[j][i] = 1
        print("fixed is ", np.shape(fixed), fixed)

        for i in range(0,64):
            fixed_sum = 0
            for j in range(0,40):
                fixed_sum += fixed[j][i]
            fixed_acc.append(fixed_sum/40)
        print("fixed accuracy is ", np.shape(fixed_acc), fixed_acc)

        # finding variable accuracy
        threshold = 0
        for l in range(50):
            print("{}th threshold is {}".format(l+1, 1-threshold))
            time = 0
            for i in range(0, 40):
                for j in range(0, 64):
                    if prediction[i][j] >= (1-threshold):
                        # print("early classified")
                        if time == 0:
                            time = j
                            if answers[i] == 1:
                                variable[l][i] = 1
                            elif answers[i] == 0:
                                variable[l][i] = 0
                            break
                    elif prediction[i][j] <= threshold:
                        # print("early classified")
                        if time == 0:
                            time = j
                            if answers[i] == 1:
                                variable[l][i] = 0
                            elif answers[i] == 0:
                                variable[l][i] = 1
                                break

                    elif j == 63:
                        time = 64
                        if prediction[i][j] >= 0.5:
                            if answers[i] == 1:
                                variable[l][i] = 1
                            elif answers[i] == 0:
                                variable[l][i] = 0
                        else:
                            if answers[i] == 1:
                                variable[l][i] = 0
                            elif answers[i] == 0:
                                variable[l][i] = 1

                times[l][i] = time
                time = 0

            threshold += 0.01
            mean_time.append(int(np.mean(times[l])))

        print("variable is ", np.shape(variable), variable)
        print("mean time is", np.shape(mean_time), mean_time)

        # finding variable accuracy for 50 threshold
        for i in range(0,50):
            variable_acc.append(np.sum(variable[i])/40)
        print(f"variable accuracy is {variable_acc}")


        # print(f"mean time is {mean_time}")
        # print(f"variable acc is {variable_acc}")
        #
        # print(f"x_axis is {x_axis}")
        # print(f"fixed acc is {fixed_acc}")

        # drawing fixed accuracy
        plt.figure(figsize=(7, 5))
        plt.plot(x_axis, fixed_acc)
        plt.title("Fixed Accuracy")
        plt.xlabel('time(token)')
        plt.ylabel('Accuracy')
        plt.ylim([0,1])
        # plt.show()
        plt.savefig(f"graph/Lee_S{subject}_{how}_fixed_acc.png")
        # plt.savefig(f"paper/Lee_S{subject}_{how}_fixed_acc.png")

        # drawing variable accuracy
        plt.figure(figsize=(7, 5))
        plt.plot(mean_time, variable_acc)
        plt.title('Variable Accuracy')
        plt.xlabel('Time(token)')
        plt.ylabel('Accuracy')
        plt.ylim([0,1])
        # plt.show()
        plt.savefig(f"graph/Lee_S{subject}_{how}_variable_acc.png")
        # plt.savefig(f"paper/Lee_S{subject}_{how}_variable_acc.png")


        # drawing both accuracy graph
        plt.figure(figsize=(7, 5))
        # plt.plot(mean_time, variable_acc, label='variable')
        plt.plot(mean_time, variable_acc, label='SPRT')
        plt.plot(x_axis, fixed_acc, label='fixed')
        plt.title('Both Accuracy')
        plt.xlabel('Time(token)')
        plt.ylabel('Accuracy')
        plt.ylim([0,1])
        plt.legend()
        # plt.show()
        plt.savefig(f"graph/Lee_S{subject}_{how}_both_acc.png")
        # plt.savefig(f"paper/Lee_S{subject}_{how}_both_acc1.png")

        return mean_time, variable_acc, x_axis, fixed_acc

    def test(self, how):

        times = []
        variable_accs = []
        fixed_accs = []
        x = []
        mean = []
        variable_accuracy = []
        for i in range(1, 55):
            time, variable, x_axis, fixed = Train(dataset='Lee').inf(subject=i, fold=1, how='fixed')
            for j in range(1, 65):
                total = 0
                total_time = 0
                # print(f"{j}: {np.where(np.array(time) == j)[0]}")
                if len(np.where(np.array(time) == j)[0]) != 0:
                    for k in np.where(np.array(time) == j)[0]:
                        total += variable[k]
                        total_time += time[k]
                    variable_accuracy.append(total / len(np.where(np.array(time) == j)[0]))
                    mean.append(int(total_time / len(np.where(np.array(time) == j)[0])))
            #         print("mean is ", mean)
            #         print("accuracy is ", variable_accuracy)
            # print(f"{i}th subject mean_time is ", np.shape(mean), mean)
            # print(f"{i}th subject variable_accuracy is ", np.shape(variable_accuracy), variable_accuracy)
            # print(f"{i}th subject time is ", np.shape(time), time)
            # print(f"{i}th subject accuracy is ", np.shape(variable), variable)
            times += time
            variable_accs += variable
            x += x_axis
            fixed_accs += fixed

        mean_time = []
        variable_acc = []
        for i in range(1, 65):
            print(f"{i}: {np.where(np.array(mean) == i)[0]}")
            summ = 0
            if len(np.where(np.array(mean) == i)[0]) != 0:
                mean_time.append(i)
                for j in np.where(np.array(mean) == i)[0]:
                    summ += variable_accuracy[j]
                variable_acc.append(summ / len(np.where(np.array(mean) == i)[0]))
        print(np.shape(mean_time), mean_time)
        print(np.shape(variable_acc), variable_acc)

        fixed_acc = []
        for i in range(1, 65):
            summ = 0
            for j in range(0, 640):
                if x[j] == i:
                    summ += fixed_accs[j]
            fixed_acc.append(summ / 10)
        print(np.shape(fixed_acc), fixed_acc)




        # times = []
        # variable_accs = []
        # fixed_accs = []
        # x = []
        # mean = []
        # variable_accuracy = []
        # for i in range(1, 11):
        #     time, variable, x_axis, fixed = Train(dataset='Lee').inf(subject=i, fold=1, how='fixed')
        #     times += time
        #     variable_accs += variable
        #     x += x_axis
        #     fixed_accs += fixed
        #
        # print(np.shape(times), times)
        # print(np.shape(variable_accs), variable_accs)
        # print(np.shape(x), x)
        # print(np.shape(fixed_accs), fixed_accs)
        #
        # mean_time = []
        # variable_acc = []
        # for i in range(1, 65):
        #     print(f"{i}: {np.where(np.array(times) == i)[0]}")
        #     summ = 0
        #     if len(np.where(np.array(times) == i)[0]) != 0:
        #         mean_time.append(i)
        #         for j in np.where(np.array(times) == i)[0]:
        #             summ += variable_accs[j]
        #         variable_acc.append(summ / len(np.where(np.array(times) == i)[0]))
        # print(np.shape(mean_time), mean_time)
        # print(np.shape(variable_acc), variable_acc)
        #
        # fixed_acc = []
        # for i in range(1, 65):
        #     summ = 0
        #     for j in range(0, 640):
        #         if x[j] == i:
        #             summ += fixed_accs[j]
        #     fixed_acc.append(summ /10)
        # print(np.shape(fixed_acc), fixed_acc)
        #
        #
        #
        #
        x_axis = [n for n in range(1, 65)]

        plt.figure(figsize=(7, 5))
        # plt.plot(mean_time, variable_acc, label='variable')
        plt.plot(mean_time, variable_acc, label='SPRT')
        plt.plot(x_axis, fixed_acc, label='fixed')
        plt.title('Both Accuracy')
        plt.xlabel('Time(token)')
        plt.ylabel('Accuracy')
        plt.ylim([0, 1])
        plt.legend()
        # plt.show()
        # plt.savefig(f"graph/Lee_S{subject}_{how}_both_acc.png")
        plt.savefig(f"paper/Lee_{how}_both_acc.png")

        print(f"time is 5 -> SPRT_acc is {variable_acc[4]}, fixed_acc is {fixed_acc[4]}")
        print(f"time is 10 -> SPRT_acc is {variable_acc[9]}, fixed_acc is {fixed_acc[9]}")
        print(f"time is 15 -> SPRT_acc is {variable_acc[14]}, fixed_acc is {fixed_acc[14]}")
        print(f"time is 20 -> SPRT_acc is {variable_acc[19]}, fixed_acc is {fixed_acc[19]}")
        print(f"time is 25 -> SPRT_acc is {variable_acc[24]}, fixed_acc is {fixed_acc[24]}")
        print(f"time is 30 -> SPRT_acc is {variable_acc[29]}, fixed_acc is {fixed_acc[29]}")
        print(f"time is 40 -> SPRT_acc is {variable_acc[39]}, fixed_acc is {fixed_acc[39]}")
        print(f"time is 50 -> SPRT_acc is {variable_acc[49]}, fixed_acc is {fixed_acc[49]}")
        print(f"time is 60 -> SPRT_acc is {variable_acc[59]}, fixed_acc is {fixed_acc[59]}")


    def data_check(self, subject):
        data = np.load(f"C:/Users/junhee/PycharmProjects/BCI/Lee_dataset/subj_{subject}_data.npy")
        label = np.load(f"C:/Users/junhee/PycharmProjects/BCI/Lee_dataset/subj_{subject}_label.npy")

        print("data is ", np.shape(data), data)
        print("label is ", np.shape(label), label)

if __name__ == '__main__':
    # wandb.login()
    # for i in range(1, 11):
    #     Train(dataset='Lee').SVtrain(subject=i, fold=1)
    Train(dataset='Lee').SVtrain(subject=1, fold=1, how="fixed")
    # Train(dataset='Lee').dep_eval(subject=1, fold=1)
    # Train(dataset='Lee').subject_dependent_category_evaluation(subject=6, fold=1)
    # Train(dataset='Lee').dep_eval(subject=6, fold=1)
    # Train(dataset='Lee').inf(subject=2, fold=1, how='fixed')
    # for i in range(11, 55):
    #     Train(dataset='Lee').inf(subject=i, fold=1, how='fixed')
    #     Train(dataset='Lee').inf(subject=i, fold=1, how='variable')
    #     # Train(dataset='Lee').SVtrain(subject=i, fold=1, how='variable')
    # Train(dataset='Lee').test(how='fixed')

    # Train(dataset='Lee').data_check(6)
    pass