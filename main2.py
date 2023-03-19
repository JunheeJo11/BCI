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

    def SVtrain(self, subject, fold):
        self.args.eval_idx = fold
        self.args.eval_subject = subject
        self.args.seq_len_hlt = 64
        # self.args.epochs = 10

        tfcv = TenFCVDataset(subject=subject, args=self.args, fold=self.args.eval_idx)
        x_train, y_train, x_val, y_val = tfcv.get_data()

        self.args.val_len = x_val.shape[0]

        train_loader = MyDataset(x_train, y_train)
        val_loader = MyDataset(x_val, y_val)

        train_iter = DataLoader(dataset=train_loader, batch_size=self.args.batch_size, shuffle=True,
                                num_workers=self.args.num_workers) #, collate_fn=self.custom_collate)
        val_iter = DataLoader(dataset=val_loader, batch_size=self.args.batch_size*4, shuffle=False,
                              num_workers=self.args.num_workers) #, collate_fn=self.custom_collate)



        trainer = Trainer(self.args)

        self.args.train_mode = 'llt'
        trainer.train(train_iter, val_iter)

        self.args.train_mode = 'hlt'
        trainer.train(train_iter, val_iter)

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

    def dep_eval(self, subject, fold):

        self.args.eval_idx = fold
        self.args.eval_subject = subject
        self.args.seq_len_hlt = 65

        tfcv = TenFCVDataset(args=self.args, subject=subject, fold=fold)
        x_val, y_val, for_test = tfcv.get_eval_data()
        self.args.seq_len_hlt = 5

        print(for_test)
        result = np.zeros((40, 28))
        accs = np.zeros((40, 28))
        y_hats = []
        step = 0
        time_accuracy = np.zeros(28)
        # threshold_accuracy = np.zeros(50)

        threshold = 0
        prob = 0
        count = 0
        threshold_accuracy = 0
        # mean_time = 0
        times = np.zeros(40)
        probs = np.zeros(40)
        mean_times = np.zeros(50)
        threshold_accuracys = np.zeros(50)

        for l in range(50):
            print("{}th threshold".format(l+1))
            print("threshold is {}".format(1-threshold))
            for k in for_test:
                idx = np.where(for_test == k)
                s_val = x_val[k-1:k, :, :]
                # print(s_val.shape)
                # print("answer is {}".format(y_val[k-1]))
                print("{}th trial, answer is {}".format(idx[0][0], y_val[k-1]))

                self.args.val_len = x_val.shape[0]
                self.args.eval_subject = subject
                j = 5
                time = 0
                while time == 0:
                    self.args.seq_len_hlt = j

                    trainer = Trainer(self.args)
                    y_hat = trainer.inference(s_val)
                    # y_hat = y_hat.cpu().detach().numpy()
                    y_hat = tuple(t.cpu() for t in y_hat)
                    # print(y_hat.shape)

                    # print(y_hat)
                    if np.max(y_hat) >= (1-threshold):
                        print("{} is bigger than {} -> early classified".format(np.max(y_hat), (1-threshold)))
                        if time == 0:
                            # print("j is {}".format(j))
                            time = j

                        if y_hat[0][0] > y_hat[0][1]:
                            prob = 0
                        else:
                            prob = 1

                        times[idx[0][0]] = time
                        probs[idx[0][0]] = prob
                        print("early classified time is {}, prob: {}".format(time, prob))
                        break

                    elif j == 32:
                        time = 32
                        if y_hat[0][0] > y_hat[0][1]:
                            prob = 0
                        else:
                            prob = 1

                        times[idx[0][0]] = time
                        probs[idx[0][0]] = prob
                        print("time is 66, prob : {}".format(prob))
                    # times[idx[0][0]] = time
                    # probs[idx[0][0]] = prob
                    j += 1

            for k in for_test:
                idx = np.where(for_test == k)
                if y_val[k-1] == probs[idx]:
                    count += 1
            print("count is {}".format(count))

            print("times : {}".format(times))
            print("probs : {}".format(probs))

            mean_time = np.average(times)
            threshold_accuracy = count/40
            print("mean_time of {}th threshold is {}".format(l, mean_time))
            print("threshold_accuracy of {}th threshold is {}".format(l, threshold_accuracy))

            mean_times[l] = mean_time
            threshold_accuracys[l] = threshold_accuracy
            count = 0
            threshold += 0.01
            times = np.zeros(40)
            probs = np.zeros(40)

        for k in for_test:
            idx = np.where(for_test == k)
            s_val = x_val[k-1:k, :, :]
            print("{}th trial".format(idx[0][0]))

            self.args.val_len = x_val.shape[0]
            self.args.eval_subject = subject

            for j in range(5, 33):
                self.args.seq_len_hlt = j

                trainer = Trainer(self.args)
                y_hat, score = trainer.inference(s_val)
                y_hat = y_hat.cpu().detach().numpy()

                max_idx = np.argmax(y_hat)
                # print("larger probability is {}".format(max_idx))
                result[step][j-5] = max_idx
                if max_idx == y_val[k-1]:
                    accs[step][j-5] = 1
                else:
                    accs[step][j-5] = 0
            step+=1

        for i in range(28):
            sum = 0
            for j in range(40):
                sum+=accs[j][i]
            time_accuracy[i] = sum/40

        print("mean times are {}".format(mean_times))
        print("threshold accuracys are {}".format(threshold_accuracys))
        print("time accuracy is {}".format(time_accuracy))

        x_axis = [5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64]
        plt.plot(mean_times, threshold_accuracys, label='threshold')
        plt.plot(x_axis, time_accuracy, label='time')
        plt.xlabel('token')
        plt.ylabel('accuracy')
        plt.legend()
        plt.show()

        return

    def inf(self):      # 랜덤한 길이의 인풋으로 학습시킨거 그래프 그릴 때 사용
        fold = 1
        subject = 6

        self.args.eval_idx = fold
        self.args.eval_subject = subject
        self.args.seq_len_hlt = 64

        tfcv = TenFCVDataset(subject=subject, args=self.args, fold=self.args.eval_idx)
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
        r = []
        # with wandb.init(project='BCI'):

        # for j in range(0,40):
        #     print(f"{j} trial")
        #     count = 0
        #     l = []
        #     r = []
        #     accuracy = []
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
        #             # print(f"length is {self.args.seq_len_hlt}", f'prediction is {yhat}', f'the answer is {y_val[0]}')
        #
        #     print("r is ", np.shape(r), r)
        #     answers.append(answer)
        # print("answers is ", np.shape(answers), answers)
        # print("left is ", np.shape(left), left)
        # print("right is ", np.shape(right), right)
        # print("prediction is ", np.shape(prediction), prediction)
        #
        # with open("answers.pkl", "wb") as f:
        #     pickle.dump(answers, f)
        # with open("left.pkl", "wb") as f:
        #     pickle.dump(left, f)
        # with open("right.pkl", "wb") as f:
        #     pickle.dump(right, f)
        # with open("prediction.pkl", "wb") as f:
        #     pickle.dump(prediction, f)

        with open('answers.pkl',"rb") as f:
            answers = pickle.load(f)
        with open('left.pkl', "rb") as f:
            left = pickle.load(f)
        with open('right.pkl',"rb") as f:
            right = pickle.load(f)
        with open('prediction.pkl',"rb") as f:
            prediction = pickle.load(f)

        print("left is ", np.shape(left), left)
        print("right is ", np.shape(right), right)
        print(len(left), len(right))
        # drawing left/right hand probability graph
        for i in range(int(len(left) / 64)):
            # plt.subplot(1,2,1)
            plt.plot(x_axis, left[i*64:(i+1)*64])
            plt.title(f"Left Hand_{i}")
            plt.xlabel('time(token)')
            plt.ylabel('prob')
            plt.ylim([0,1])
            # wandb.log({"Left Hand": wandb.Image(plt)})
            plt.show()
        # for i in range(int(len(right)/64)):
        #     # plt.subplot(1,2,2)
        #     plt.plot(x_axis, right[i*64:(i+1)*64])
        #     plt.title(f"Right Hand_{i}")
        #     plt.xlabel('time(token)')
        #     plt.ylabel('prob')
        #     plt.ylim([0, 1])
        #     # wandb.log({"Right Hand": wandb.Image(plt)})
        #     plt.show()


        # finding fixed accuracy
        # for i in range(0,64):
        #     for j in range(0,40):
        #         if prediction[j][i] > 0.5:
        #             if answers[j] == 1:
        #                 fixed[j][i] = 1
        #             elif answers[j] == 0:
        #                 fixed[j][i] = 0
        #         elif prediction[j][i] < 0.5:
        #             if answers[j] == 1:
        #                 fixed[j][i] = 0
        #             elif answers[j] == 0:
        #                 fixed[j][i] = 1
        # print("fixed is ", np.shape(fixed), fixed)
        #
        # for i in range(0,64):
        #     fixed_sum = 0
        #     for j in range(0,40):
        #         fixed_sum += fixed[j][i]
        #     fixed_acc.append(fixed_sum/40)
        # print("fixed accuracy is ", np.shape(fixed_acc), fixed_acc)

        # finding variable accuracy
        # threshold = 0
        # for l in range(50):
        #     print("{}th threshold is {}".format(l+1, 1-threshold))
        #     time = 0
        #     for i in range(0, 40):
        #         for j in range(0, 64):
        #             if prediction[i][j] >= (1-threshold):
        #                 # print("early classified")
        #                 if time == 0:
        #                     time = j
        #                     if answers[i] == 1:
        #                         variable[l][i] = 1
        #                     elif answers[i] == 0:
        #                         variable[l][i] = 0
        #                     break
        #             elif prediction[i][j] <= threshold:
        #                 # print("early classified")
        #                 if time == 0:
        #                     time = j
        #                     if answers[i] == 1:
        #                         variable[l][i] = 0
        #                     elif answers[i] == 0:
        #                         variable[l][i] = 1
        #                         break
        #
        #             elif j == 63:
        #                 time = 64
        #                 if prediction[i][j] >= 0.5:
        #                     if answers[i] == 1:
        #                         variable[l][i] = 1
        #                     elif answers[i] == 0:
        #                         variable[l][i] = 0
        #                 else:
        #                     if answers[i] == 1:
        #                         variable[l][i] = 0
        #                     elif answers[i] == 0:
        #                         variable[l][i] = 1
        #
        #         times[l][i] = time
        #         time = 0
        #
        #     threshold += 0.01
        #     mean_time.append(np.mean(times[l]))
        #
        # print("variable is ", np.shape(variable), variable)
        # print("mean time is", np.shape(mean_time), mean_time)

        # finding variable accuracy for 50 threshold
        # for i in range(0,50):
        #     variable_acc.append(np.sum(variable[i])/40)
        # print(f"variable accuracy is {variable_acc}")




        # plt.plot(x_axis, fixed_acc)
        # plt.title("Fixed Accuracy")
        # plt.xlabel('time(token)')
        # plt.ylabel('Accuracy')
        # plt.ylim([0,1])
        # plt.show()

        # variable accuracy
        # plt.plot(mean_time, variable_acc)
        # plt.title('Variable Accuracy')
        # plt.xlabel('Time(token)')
        # plt.ylabel('Accuracy')
        # plt.ylim([0,1])
        # plt.show()


        # both accuracy graph

        # plt.plot(mean_time, variable_acc, label='variable')
        # plt.plot(x_axis, fixed_acc, label='fixed')
        # plt.xlabel('Time(token)')
        # plt.ylabel('Accuracy')
        # plt.ylim([0,1])
        # plt.legend()
        # plt.show()


        # print(left)
        # print(right)
        # left = [0.04497891664505005, 0.045045651495456696, 0.045060571283102036, 0.045103903859853745, 0.045188840478658676, 0.04525015875697136, 0.0449632927775383, 0.044815801084041595, 0.044757165014743805, 0.04472650587558746, 0.04471433162689209, 0.04470525681972504, 0.044697415083646774, 0.04468664526939392, 0.04467744752764702, 0.04467049986124039, 0.04466371610760689, 0.04465848580002785, 0.04465342313051224, 0.044649314135313034, 0.0446472205221653, 0.044645410031080246, 0.044642236083745956, 0.04463756084442139, 0.044632572680711746, 0.044629666954278946, 0.04462773725390434, 0.04462629556655884, 0.044625621289014816, 0.044624291360378265, 0.04462188482284546, 0.04462084174156189, 0.04462098702788353, 0.04462151229381561, 0.04462060704827309, 0.0446200929582119, 0.044620268046855927, 0.044620539993047714, 0.04461999237537384, 0.04461835324764252, 0.04461708664894104, 0.0446147695183754, 0.044612206518650055, 0.04461048170924187, 0.044608600437641144, 0.04460688307881355, 0.04460521787405014, 0.044603753834962845, 0.04460395127534866, 0.044605135917663574, 0.04460608959197998, 0.044607218354940414, 0.04460810124874115, 0.04460861161351204, 0.04460820183157921, 0.044605497270822525, 0.04460274055600166, 0.044600557535886765, 0.04459915682673454, 0.04459849372506142, 0.04459848254919052, 0.044598694890737534, 0.044598326086997986, 0.044597577303647995, 0.0443095862865448, 0.04427941516041756, 0.044272299855947495, 0.04426063969731331, 0.04426020383834839, 0.04426201805472374, 0.044251810759305954, 0.04424687847495079, 0.04424305632710457, 0.04423927888274193, 0.04423022270202637, 0.04422364383935928, 0.04421422258019447, 0.04420439153909683, 0.04420890659093857, 0.044208962470293045, 0.04420368745923042, 0.04419967532157898, 0.04419467970728874, 0.044187143445014954, 0.0441846139729023, 0.04418426752090454, 0.04418418928980827, 0.044182050973176956, 0.044181130826473236, 0.04418100789189339, 0.044181156903505325, 0.044180624186992645, 0.04417925328016281, 0.044181276112794876, 0.04418116435408592, 0.04418283700942993, 0.044184450060129166, 0.044185820966959, 0.04418643191456795, 0.0441865436732769, 0.044186364859342575, 0.04418715462088585, 0.04418804496526718, 0.04418766126036644, 0.04418649524450302, 0.044184066355228424, 0.044181860983371735, 0.04418018087744713, 0.04417794570326805, 0.044174160808324814, 0.04417131468653679, 0.044167112559080124, 0.044161632657051086, 0.04415861517190933, 0.044152963906526566, 0.044151898473501205, 0.0441519170999527, 0.04415182024240494, 0.04415089264512062, 0.04415101557970047, 0.04415165260434151, 0.04415237158536911, 0.04415342956781387, 0.04415498301386833, 0.04415634274482727, 0.04415731132030487, 0.04415912181138992, 0.04416054114699364, 0.959168553352356, 0.9593061208724976, 0.9596688151359558, 0.9595398306846619, 0.9593431949615479, 0.9593502879142761, 0.959457516670227, 0.959440290927887, 0.9595018625259399, 0.9594882130622864, 0.9592775702476501, 0.958968997001648, 0.9588488340377808, 0.9587934017181396, 0.9577116966247559, 0.9495888352394104, 0.9325041770935059, 0.9137883186340332, 0.9208036661148071, 0.9311423897743225, 0.9361483454704285, 0.9383928179740906, 0.9414909482002258, 0.39076560735702515, 0.051445599645376205, 0.04777570441365242, 0.04713493213057518, 0.047035519033670425, 0.047137875109910965, 0.04726903885602951, 0.047346070408821106, 0.04746183753013611, 0.04762416332960129, 0.047754917293787, 0.04786888509988785, 0.04798652604222298, 0.04810772091150284, 0.04827210679650307, 0.048413049429655075, 0.04853694140911102, 0.048643194139003754, 0.04870474338531494, 0.0487358532845974, 0.04871908575296402, 0.048720113933086395, 0.0487610325217247, 0.04879368469119072, 0.04875488951802254, 0.048790886998176575, 0.048935335129499435, 0.04910452291369438, 0.04928530380129814, 0.049459800124168396, 0.04954089969396591, 0.049593694508075714, 0.04968389868736267, 0.04984748363494873, 0.05003827065229416, 0.05023615434765816, 0.050539471209049225, 0.0510379858314991, 0.0514976903796196, 0.051921065896749496, 0.05198351666331291, 0.04492129012942314, 0.04492693766951561, 0.044995736330747604, 0.045029956847429276, 0.044969718903303146, 0.0448080450296402, 0.04480845853686333, 0.04483429715037346, 0.044877614825963974, 0.04491061344742775, 0.04492795094847679, 0.04495519772171974, 0.044982172548770905, 0.04497214034199715, 0.04497811943292618, 0.04498870670795441, 0.04499311372637749, 0.0450022891163826, 0.04501313343644142, 0.04502643644809723, 0.045036524534225464, 0.0450441800057888, 0.04505189135670662, 0.04505769908428192, 0.045042604207992554, 0.04500536993145943, 0.04497545212507248, 0.0449666790664196, 0.04495544359087944, 0.04494158923625946, 0.044929783791303635, 0.04491795599460602, 0.04491325095295906, 0.04491245001554489, 0.044913530349731445, 0.04491543769836426, 0.04491627961397171, 0.044916585087776184, 0.044916197657585144, 0.04491779953241348, 0.0449194498360157, 0.04491967335343361, 0.04492086172103882, 0.04492196440696716, 0.0449238047003746, 0.0449252687394619, 0.044924020767211914, 0.044923774898052216, 0.04492499306797981, 0.0449257493019104, 0.04492137208580971, 0.04491722211241722, 0.04491788148880005, 0.04491714388132095, 0.04491812735795975, 0.04492080211639404, 0.04492149502038956, 0.044918205589056015, 0.04491523653268814, 0.04491433501243591, 0.044912394136190414, 0.044909510761499405, 0.044907502830028534, 0.04490486532449722, 0.9462569952011108, 0.9149768352508545, 0.9439936280250549, 0.9549234509468079, 0.955636203289032, 0.9566038250923157, 0.9574071168899536, 0.9574429988861084, 0.9576123952865601, 0.9574234485626221, 0.9573382139205933, 0.9573794007301331, 0.9572163820266724, 0.9566488862037659, 0.9557821154594421, 0.9555757641792297, 0.9554170966148376, 0.9549214839935303, 0.9545027017593384, 0.9541769027709961, 0.9539427161216736, 0.9537948966026306, 0.953679621219635, 0.9535568952560425, 0.9534147381782532, 0.953307032585144, 0.9532255530357361, 0.9531340599060059, 0.9530899524688721, 0.9532609581947327, 0.9534661173820496, 0.9535694718360901, 0.9534446597099304, 0.9517545104026794, 0.944916844367981, 0.9338042140007019, 0.9054538607597351, 0.876300036907196, 0.2855229675769806, 0.06311925500631332, 0.053912848234176636, 0.04903627932071686, 0.04740692302584648, 0.04684242233633995, 0.046469736844301224, 0.04709189012646675, 0.04741000756621361, 0.04753883555531502, 0.046131499111652374, 0.04574824869632721, 0.04574687406420708, 0.046029482036828995, 0.04664968326687813, 0.04767291992902756, 0.04945443943142891, 0.05110280215740204, 0.05871807411313057, 0.07545246183872223, 0.09097492694854736, 0.11967131495475769, 0.18362170457839966, 0.2007131725549698, 0.1293010413646698, 0.12845267355442047, 0.9571138620376587, 0.9488738179206848, 0.9181033968925476, 0.9097428917884827, 0.9158728718757629, 0.9283791184425354, 0.9381116032600403, 0.9332523941993713, 0.6514257788658142, 0.08082661032676697, 0.0527866929769516, 0.047955989837646484, 0.04758560284972191, 0.04789300635457039, 0.04748741537332535, 0.04758359119296074, 0.04701017588376999, 0.04581840708851814, 0.04554890841245651, 0.04538021236658096, 0.04507066681981087, 0.04478462412953377, 0.04460472986102104, 0.04443857818841934, 0.04435139149427414, 0.04427226260304451, 0.04420764371752739, 0.04417569935321808, 0.044159743934869766, 0.04414543882012367, 0.044120434671640396, 0.04408896341919899, 0.04407115280628204, 0.04406552389264107, 0.04406093806028366, 0.04407161474227905, 0.044092852622270584, 0.04410865530371666, 0.044132620096206665, 0.04416171833872795, 0.04417957738041878, 0.04416298493742943, 0.04412855952978134, 0.04412690922617912, 0.04417401924729347, 0.04421178624033928, 0.04424506798386574, 0.04427919536828995, 0.044315461069345474, 0.04435533285140991, 0.04441402480006218, 0.04451899603009224, 0.044616568833589554, 0.044678058475255966, 0.04473830386996269, 0.04480075463652611, 0.044880229979753494, 0.04495759308338165, 0.044996533542871475, 0.045026808977127075, 0.045049164444208145, 0.04504867270588875, 0.04505829140543938, 0.04508250579237938, 0.04399586096405983, 0.044083546847105026, 0.04409411922097206, 0.044081252068281174, 0.04407954588532448, 0.04405944421887398, 0.044050101190805435, 0.04407621547579765, 0.04412167891860008, 0.04414376989006996, 0.04414483532309532, 0.04413771256804466, 0.04413044825196266, 0.0441267304122448, 0.04413218796253204, 0.04413967207074165, 0.04415129870176315, 0.044159404933452606, 0.04416812211275101, 0.04417659342288971, 0.04418124631047249, 0.04418237879872322, 0.04418618977069855, 0.04419126734137535, 0.04419465735554695, 0.04420091584324837, 0.044209208339452744, 0.04421434551477432, 0.0442182756960392, 0.04422460123896599, 0.04422806203365326, 0.04422944784164429, 0.04423050582408905, 0.04423099383711815, 0.04423188045620918, 0.04423228278756142, 0.04423167556524277, 0.044232338666915894, 0.04423478990793228, 0.04423387348651886, 0.044232726097106934, 0.04423163831233978, 0.0442308709025383, 0.044230375438928604, 0.044230811297893524, 0.04423198103904724, 0.0442340262234211, 0.04423736035823822, 0.044242359697818756, 0.044249214231967926, 0.04425659775733948, 0.044263873249292374, 0.044270195066928864, 0.044275928288698196, 0.04428177326917648, 0.04428758844733238, 0.04428968206048012, 0.04428982734680176, 0.04428873583674431, 0.044289425015449524, 0.04429037496447563, 0.04429011419415474, 0.04429011419415474, 0.04429033398628235, 0.045660220086574554, 0.04571695625782013, 0.04580563306808472, 0.04588295891880989, 0.04590648412704468, 0.045860275626182556, 0.04577113315463066, 0.04570373147726059, 0.04565732181072235, 0.04549739882349968, 0.04530603811144829, 0.045220986008644104, 0.04517758637666702, 0.045133013278245926, 0.04508950561285019, 0.04505639895796776, 0.04502924531698227, 0.04500741511583328, 0.0449787899851799, 0.044947605580091476, 0.04492293670773506, 0.04491286352276802, 0.04490942507982254, 0.0449075773358345, 0.044907063245773315, 0.04490493983030319, 0.04490014165639877, 0.04489042982459068, 0.044877104461193085, 0.04486439749598503, 0.04485530033707619, 0.044854745268821716, 0.04485137760639191, 0.04485100507736206, 0.044849324971437454, 0.044846247881650925, 0.044843174517154694, 0.04483874887228012, 0.04483472928404808, 0.044830720871686935, 0.044825807213783264, 0.04482123255729675, 0.0448172427713871, 0.044812507927417755, 0.04480655863881111, 0.04480021074414253, 0.044798802584409714, 0.04480213299393654, 0.04480515047907829, 0.044807229191064835, 0.044807277619838715, 0.04480533301830292, 0.04480347782373428, 0.044803641736507416, 0.04480137676000595, 0.04479757323861122, 0.04479413107037544, 0.0447901114821434, 0.04478833079338074, 0.04479100927710533, 0.04479379579424858, 0.04479784518480301, 0.04479771852493286, 0.04479570314288139, 0.0443505235016346, 0.04421902075409889, 0.04412475600838661, 0.04417926445603371, 0.044113073498010635, 0.04392369091510773, 0.04381632059812546, 0.04370447248220444, 0.04357970505952835, 0.04351799562573433, 0.043457675725221634, 0.04341937601566315, 0.043468959629535675, 0.043494127690792084, 0.04351396858692169, 0.043540362268686295, 0.04355904087424278, 0.04352942109107971, 0.04351132735610008, 0.043498195707798004, 0.04347330331802368, 0.04344937205314636, 0.04343119263648987, 0.04341714084148407, 0.0434124581515789, 0.043409574776887894, 0.04340805858373642, 0.043410588055849075, 0.0434105321764946, 0.04340500384569168, 0.04339635372161865, 0.043387021869421005, 0.043382350355386734, 0.04337847977876663, 0.04337620362639427, 0.043373871594667435, 0.04337509348988533, 0.04338892176747322, 0.043394215404987335, 0.04340019077062607, 0.04340333864092827, 0.04340608790516853, 0.04340613633394241, 0.04340652748942375, 0.043406806886196136, 0.043407224118709564, 0.04341096803545952, 0.04341451823711395, 0.04342031106352806, 0.043423183262348175, 0.043421678245067596, 0.043420031666755676, 0.04341743886470795, 0.04341316968202591, 0.043408215045928955, 0.04340328276157379, 0.04339827597141266, 0.04339592158794403, 0.04339461028575897, 0.043392766267061234, 0.04339012876152992, 0.04338609054684639, 0.04338301345705986, 0.04338184744119644, 0.045543819665908813, 0.04639751464128494, 0.05390895903110504, 0.05856805667281151, 0.05871553346514702, 0.059359945356845856, 0.05753013491630554, 0.05495946481823921, 0.05477740988135338, 0.053747549653053284, 0.050671037286520004, 0.046625010669231415, 0.046042706817388535, 0.04569322243332863, 0.04544079676270485, 0.04525715112686157, 0.04512446001172066, 0.04511028900742531, 0.045311037451028824, 0.04610615596175194, 0.04636064171791077, 0.04636526480317116, 0.046452030539512634, 0.04652559757232666, 0.04659181088209152, 0.046914927661418915, 0.04726152494549751, 0.04711511358618736, 0.0466841496527195, 0.04686513915657997, 0.046818334609270096, 0.04666656628251076, 0.04652950167655945, 0.04643884301185608, 0.04637084901332855, 0.046310894191265106, 0.04626163840293884, 0.046244196593761444, 0.04556258022785187, 0.04559706524014473, 0.04556921869516373, 0.045531265437603, 0.04552942514419556, 0.0456705316901207, 0.04598619416356087, 0.04662463068962097, 0.04777633771300316, 0.04704945534467697, 0.04605011269450188, 0.04627365246415138, 0.04662196710705757, 0.04736292362213135, 0.04815792664885521, 0.048638541251420975, 0.049140337854623795, 0.04940799996256828, 0.04881540685892105, 0.04827079549431801, 0.04831325635313988, 0.04873663932085037, 0.04922381043434143, 0.04944813624024391, 0.04985266178846359, 0.05023294687271118, 0.9473430514335632, 0.9473538398742676, 0.9493798017501831, 0.9528464078903198, 0.9512224197387695, 0.9240275621414185, 0.7023439407348633, 0.30155032873153687, 0.15698836743831635, 0.10452854633331299, 0.08371955901384354, 0.0707443356513977, 0.06268592923879623, 0.05487782880663872, 0.048964597284793854, 0.044657617807388306, 0.04506571963429451, 0.04396633803844452, 0.042786501348018646, 0.04235090687870979, 0.04221166670322418, 0.04225732386112213, 0.042443517595529556, 0.04253892973065376, 0.04193085804581642, 0.04335314407944679, 0.04366757720708847, 0.04367474839091301, 0.04361080005764961, 0.04354805871844292, 0.04349339380860329, 0.04344262555241585, 0.043397996574640274, 0.04336285591125488, 0.04332811385393143, 0.04329656437039375, 0.04327092319726944, 0.04324668273329735, 0.043220385909080505, 0.04319605231285095, 0.043178245425224304, 0.04316197708249092, 0.04314655438065529, 0.04313000291585922, 0.04311523586511612, 0.04310376197099686, 0.04309255629777908, 0.04308225214481354, 0.04307295009493828, 0.04306359961628914, 0.04305490478873253, 0.04304103925824165, 0.043021999299526215, 0.04300548881292343, 0.042986396700143814, 0.042967867106199265, 0.04294787347316742, 0.04292141646146774, 0.04288887232542038, 0.042853519320487976, 0.042822275310754776, 0.04278876632452011, 0.04275498166680336, 0.04272465780377388, 0.9318732023239136, 0.9301241040229797, 0.9299131631851196, 0.9308975338935852, 0.9275144934654236, 0.8547015190124512, 0.06950747221708298, 0.08063490688800812, 0.06754377484321594, 0.05308939144015312, 0.0536547489464283, 0.05396704003214836, 0.05283220484852791, 0.047450706362724304, 0.04511965811252594, 0.04467331990599632, 0.044552210718393326, 0.044548362493515015, 0.04455956071615219, 0.044578667730093, 0.04458973556756973, 0.04459753632545471, 0.04459879919886589, 0.04459121823310852, 0.04457435384392738, 0.04455909505486488, 0.04454849287867546, 0.04453151673078537, 0.044508323073387146, 0.044484302401542664, 0.04447197914123535, 0.044465720653533936, 0.04445433244109154, 0.04443623498082161, 0.04442021995782852, 0.0444079153239727, 0.044400207698345184, 0.04439692944288254, 0.04439551383256912, 0.04439059644937515, 0.04438373073935509, 0.04439634084701538, 0.04440837725996971, 0.04439014941453934, 0.04437541961669922, 0.04437132552266121, 0.04436752200126648, 0.04437083750963211, 0.044381920248270035, 0.04439454525709152, 0.04440564662218094, 0.04440855234861374, 0.044413380324840546, 0.04442449286580086, 0.044429097324609756, 0.04442537948489189, 0.04441934823989868, 0.044344983994960785, 0.044275302439928055, 0.044235244393348694, 0.04421337693929672, 0.04420849680900574, 0.04420571029186249, 0.04420533403754234, 0.9587844014167786, 0.9588438272476196, 0.9589534997940063, 0.9590616226196289, 0.9589959383010864, 0.9589318037033081, 0.9588911533355713, 0.9588248133659363, 0.9587612748146057, 0.9585208296775818, 0.9579166173934937, 0.9573845267295837, 0.9568772315979004, 0.956407904624939, 0.9559280276298523, 0.9553449749946594, 0.9545930624008179, 0.9537088871002197, 0.9527278542518616, 0.951590895652771, 0.9500041604042053, 0.9445019364356995, 0.9085911512374878, 0.8566875457763672, 0.7999095916748047, 0.7598251700401306, 0.738807737827301, 0.7293928265571594, 0.7191388607025146, 0.7061287760734558, 0.6877964735031128, 0.6902043223381042, 0.6900641322135925, 0.6719920635223389, 0.6560768485069275, 0.6471120715141296, 0.6226142048835754, 0.5901601910591125, 0.550539493560791, 0.5144475698471069, 0.47538965940475464, 0.4562302529811859, 0.3823648989200592, 0.24020804464817047, 0.17736700177192688, 0.13370786607265472, 0.10194005817174911, 0.08318159729242325, 0.07228372246026993, 0.0652465969324112, 0.060346074402332306, 0.05677609518170357, 0.0549711175262928, 0.05418475344777107, 0.05370402708649635, 0.0534549243748188, 0.05335837975144386, 0.05319040268659592, 0.05310114100575447, 0.05315640941262245, 0.05328305438160896, 0.05373787879943848, 0.06152277812361717, 0.0726398304104805, 0.9294523596763611, 0.9494611620903015, 0.953001856803894, 0.9542739391326904, 0.9559577107429504, 0.9563855528831482, 0.9560573697090149, 0.9558730721473694, 0.9556075930595398, 0.9554176330566406, 0.9553704261779785, 0.9554476737976074, 0.9555119872093201, 0.9557269811630249, 0.9557689428329468, 0.955604612827301, 0.9556578397750854, 0.9559367895126343, 0.9559923410415649, 0.9559410214424133, 0.9558215141296387, 0.955696165561676, 0.9555438756942749, 0.9552010297775269, 0.955000102519989, 0.9550298452377319, 0.955026388168335, 0.954951822757721, 0.954935610294342, 0.9548787474632263, 0.9546326398849487, 0.9544169306755066, 0.9541837573051453, 0.953766405582428, 0.9532972574234009, 0.9527354836463928, 0.9516732692718506, 0.9493686556816101, 0.9463432431221008, 0.9442154169082642, 0.9397491812705994, 0.9340860247612, 0.9337949752807617, 0.9318181276321411, 0.926019012928009, 0.9065523147583008, 0.884327232837677, 0.8746391534805298, 0.8447556495666504, 0.8069677352905273, 0.7855530381202698, 0.7518690824508667, 0.644699215888977, 0.45053738355636597, 0.25528669357299805, 0.1460256278514862, 0.07508402317762375, 0.05782044306397438, 0.053471751511096954, 0.051370587199926376, 0.050608593970537186, 0.0503133125603199, 0.049788303673267365, 0.04917306825518608, 0.04458154737949371, 0.04463396221399307, 0.04461495578289032, 0.044593583792448044, 0.04452483728528023, 0.044461265206336975, 0.04439660534262657, 0.04433344677090645, 0.04428720474243164, 0.044253282248973846, 0.044252846390008926, 0.04425365850329399, 0.044255126267671585, 0.04425244778394699, 0.04424062371253967, 0.04422484338283539, 0.04420942813158035, 0.04419894143939018, 0.04418843984603882, 0.04418081417679787, 0.04418690875172615, 0.044188328087329865, 0.044190723448991776, 0.04419909417629242, 0.04420918598771095, 0.04421834275126457, 0.044221799820661545, 0.04422248527407646, 0.044222641736269, 0.044222641736269, 0.04422270879149437, 0.044223666191101074, 0.04422631859779358, 0.044228218495845795, 0.044235050678253174, 0.044252656400203705, 0.0442778617143631, 0.044308342039585114, 0.04433610290288925, 0.04436608776450157, 0.044387560337781906, 0.04439666494727135, 0.044401563704013824, 0.04440394788980484, 0.04440586268901825, 0.044408299028873444, 0.04441189765930176, 0.04441480338573456, 0.04441629350185394, 0.04441710188984871, 0.04441814497113228, 0.04442007839679718, 0.04442254453897476, 0.04442483186721802, 0.04442671313881874, 0.044429052621126175, 0.04443136975169182, 0.04443440958857536, 0.044437214732170105, 0.044440463185310364, 0.04444427043199539, 0.044448014348745346, 0.044452812522649765, 0.04446074366569519, 0.9509081244468689, 0.9479456543922424, 0.9497044086456299, 0.9510769248008728, 0.9502473473548889, 0.9495318531990051, 0.9486283659934998, 0.9438035488128662, 0.9203383326530457, 0.8585466146469116, 0.7350966930389404, 0.4698755741119385, 0.20510827004909515, 0.12073531746864319, 0.08762576431035995, 0.074307382106781, 0.059572670608758926, 0.05516890436410904, 0.0538046695291996, 0.0528336837887764, 0.05193248763680458, 0.05117397755384445, 0.05072632431983948, 0.050364330410957336, 0.04943644627928734, 0.04921939596533775, 0.04901688173413277, 0.04890482500195503, 0.048809558153152466, 0.04865480586886406, 0.0484575554728508, 0.048304975032806396, 0.048181913793087006, 0.04807177186012268, 0.04798971116542816, 0.0479319766163826, 0.04784224182367325, 0.04774760082364082, 0.04767283797264099, 0.047606032341718674, 0.047562163323163986, 0.047557391226291656, 0.047561660408973694, 0.04754223674535751, 0.047545790672302246, 0.0475723035633564, 0.047615159302949905, 0.04764912277460098, 0.047649528831243515, 0.0476444810628891, 0.047639813274145126, 0.0476163886487484, 0.047590360045433044, 0.04756270349025726, 0.04753386974334717, 0.04750356078147888, 0.04746261239051819, 0.04742354527115822, 0.04738619178533554, 0.04735254496335983, 0.04732454568147659, 0.04729593172669411, 0.04727141186594963, 0.047245193272829056, 0.04895630478858948, 0.05100608989596367, 0.0533982589840889, 0.08685386180877686, 0.1105199083685875, 0.05896701291203499, 0.04523107036948204, 0.044983625411987305, 0.04486984759569168, 0.044807929545640945, 0.04475625976920128, 0.044721819460392, 0.044684361666440964, 0.04466739296913147, 0.044665608555078506, 0.04465962573885918, 0.04464484751224518, 0.0447029173374176, 0.0447213314473629, 0.044689543545246124, 0.044664621353149414, 0.044646911323070526, 0.044621292501688004, 0.04459666460752487, 0.04457515850663185, 0.044556643813848495, 0.04454008862376213, 0.04452033340930939, 0.04450128972530365, 0.044490888714790344, 0.04448241740465164, 0.044470079243183136, 0.04446292296051979, 0.044461723417043686, 0.044460147619247437, 0.04445502161979675, 0.04445040225982666, 0.044448480010032654, 0.04444722831249237, 0.044448982924222946, 0.04444942995905876, 0.04444686323404312, 0.044443279504776, 0.044437650591135025, 0.044433631002902985, 0.04443088546395302, 0.04442719370126724, 0.044423703104257584, 0.0444209985435009, 0.0444178469479084, 0.044414106756448746, 0.04440963268280029, 0.0444050058722496, 0.0444009006023407, 0.04439583420753479, 0.04439057409763336, 0.04438519477844238, 0.04437955096364021, 0.04437392204999924, 0.04436644911766052, 0.04435998201370239, 0.04435579851269722, 0.04435037821531296, 0.04434667155146599, 0.04798922687768936, 0.04756574705243111, 0.04884318262338638, 0.049927446991205215, 0.04987947270274162, 0.048850491642951965, 0.04830923303961754, 0.04794839769601822, 0.047573115676641464, 0.04728376865386963, 0.04687149077653885, 0.04660802707076073, 0.0464542955160141, 0.046239279210567474, 0.04602621868252754, 0.045841068029403687, 0.04565450921654701, 0.04546670988202095, 0.0452347993850708, 0.04498983174562454, 0.04478378966450691, 0.04458904638886452, 0.04441610723733902, 0.04430690035223961, 0.044262707233428955, 0.044203080236911774, 0.04414227604866028, 0.04413115233182907, 0.044171541929244995, 0.044154249131679535, 0.04412013664841652, 0.04409152641892433, 0.04406772926449776, 0.044047821313142776, 0.04403237625956535, 0.044021785259246826, 0.0440203957259655, 0.04402546584606171, 0.04403315857052803, 0.04403864964842796, 0.04403337463736534, 0.044026605784893036, 0.044022273272275925, 0.044017188251018524, 0.04401431605219841, 0.04401198774576187, 0.04401715472340584, 0.044028524309396744, 0.04404169321060181, 0.04405510425567627, 0.04407569393515587, 0.04409673810005188, 0.044119831174612045, 0.044138818979263306, 0.04414726421236992, 0.04415053129196167, 0.044152241200208664, 0.04415391758084297, 0.0441548116505146, 0.04415857046842575, 0.04416033253073692, 0.044161923229694366, 0.04416383057832718, 0.0441637821495533, 0.06839941442012787, 0.3199765682220459, 0.11943913251161575, 0.2166604995727539, 0.49615368247032166, 0.12421464920043945, 0.04901037737727165, 0.04684623330831528, 0.046618856489658356, 0.046594277024269104, 0.04703661799430847, 0.04746684432029724, 0.04756840690970421, 0.04785604402422905, 0.04835757240653038, 0.048824068158864975, 0.04924893379211426, 0.04947424307465553, 0.04967919737100601, 0.0500003919005394, 0.050179753452539444, 0.05027049407362938, 0.049853116273880005, 0.048430487513542175, 0.047422025352716446, 0.047169122844934464, 0.04695434495806694, 0.046734441071748734, 0.04647429659962654, 0.046288955956697464, 0.04627405107021332, 0.04622424393892288, 0.04619256407022476, 0.046207912266254425, 0.04616611450910568, 0.04605035483837128, 0.046005040407180786, 0.04597659036517143, 0.045944180339574814, 0.045915380120277405, 0.04585210978984833, 0.04579387232661247, 0.045710060745477676, 0.045639269053936005, 0.04554762318730354, 0.04543141648173332, 0.0453401543200016, 0.045250795781612396, 0.0451706163585186, 0.045096490532159805, 0.04499739408493042, 0.04496736451983452, 0.04495612531900406, 0.0449431873857975, 0.04493258148431778, 0.044882070273160934, 0.04479901120066643, 0.04472418874502182, 0.044649749994277954, 0.04453672096133232, 0.044522035866975784, 0.04449992626905441, 0.044491253793239594, 0.04448696970939636, 0.8712580800056458, 0.6737424731254578, 0.07975040376186371, 0.13678759336471558, 0.1040760949254036, 0.08031679689884186, 0.06546411663293839, 0.054678887128829956, 0.04883811995387077, 0.045935314148664474, 0.04467417299747467, 0.04401642456650734, 0.04362599551677704, 0.04332884028553963, 0.04303364083170891, 0.04285449534654617, 0.04271828010678291, 0.04257940500974655, 0.042432915419340134, 0.042287882417440414, 0.04216645285487175, 0.04208315163850784, 0.04198972508311272, 0.041907548904418945, 0.04188746213912964, 0.0418802872300148, 0.04187403991818428, 0.041868291795253754, 0.04186122864484787, 0.041855111718177795, 0.041850753128528595, 0.0418458990752697, 0.04184700548648834, 0.04185546562075615, 0.0418669730424881, 0.04187345877289772, 0.0418742410838604, 0.041870128363370895, 0.04186222702264786, 0.041852034628391266, 0.041842009872198105, 0.04183358699083328, 0.041826993227005005, 0.041822563856840134, 0.041818879544734955, 0.04181598871946335, 0.041815441101789474, 0.04181864857673645, 0.041827768087387085, 0.04184257239103317, 0.041857942938804626, 0.04187008738517761, 0.04187441244721413, 0.041871245950460434, 0.04186417907476425, 0.041848134249448776, 0.041824012994766235, 0.041801683604717255, 0.041775673627853394, 0.04175134375691414, 0.04173089563846588, 0.04170958325266838, 0.041681863367557526, 0.04166007786989212, 0.9527446627616882, 0.951945424079895, 0.9515538811683655, 0.9514632225036621, 0.9510305523872375, 0.9504324197769165, 0.9496753811836243, 0.9487075805664062, 0.9469006657600403, 0.9433497190475464, 0.937179684638977, 0.9231706261634827, 0.8944230675697327, 0.8360985517501831, 0.7625455260276794, 0.6683632135391235, 0.373508483171463, 0.11115539073944092, 0.04655725136399269, 0.04546624422073364, 0.04503535479307175, 0.04484634846448898, 0.04476804658770561, 0.04470697417855263, 0.044667091220617294, 0.04463621601462364, 0.04460504278540611, 0.04458014667034149, 0.04456644132733345, 0.04456186294555664, 0.04456145688891411, 0.04456309229135513, 0.04456188902258873, 0.044557590037584305, 0.044553183019161224, 0.04455085098743439, 0.04454902559518814, 0.04454691335558891, 0.04454537108540535, 0.04454497992992401, 0.044544517993927, 0.04454445466399193, 0.04454643279314041, 0.04454861953854561, 0.0445510670542717, 0.044551946222782135, 0.04455014690756798, 0.04454902559518814, 0.04454774409532547, 0.04454391822218895, 0.04454132914543152, 0.04454311728477478, 0.044544730335474014, 0.04454588517546654, 0.04454697296023369, 0.04454835131764412, 0.04455198347568512, 0.044553738087415695, 0.04455342888832092, 0.044554829597473145, 0.04455873370170593, 0.044559698551893234, 0.044558845460414886, 0.04455651342868805]
        # right = [0.9588198065757751, 0.9586390256881714, 0.9584578275680542, 0.9587478041648865, 0.9592548608779907, 0.9597657322883606, 0.959757387638092, 0.8879106044769287, 0.9545325040817261, 0.9578434228897095, 0.9586688280105591, 0.9588034749031067, 0.9588325023651123, 0.9588245153427124, 0.9589197635650635, 0.9590432047843933, 0.9591110944747925, 0.959093451499939, 0.9589699506759644, 0.9589835405349731, 0.9590770602226257, 0.9590989947319031, 0.9590834379196167, 0.9591192007064819, 0.9592073559761047, 0.9592600464820862, 0.9592859148979187, 0.9592984318733215, 0.9592963457107544, 0.9592862129211426, 0.9593123197555542, 0.9592959880828857, 0.9592163562774658, 0.9592432379722595, 0.9592262506484985, 0.9595407843589783, 0.9593557715415955, 0.9592704176902771, 0.9597944021224976, 0.9608018398284912, 0.9610469937324524, 0.9610838294029236, 0.9610627293586731, 0.9610612988471985, 0.9610738158226013, 0.9610665440559387, 0.9610707759857178, 0.9365975856781006, 0.9363828301429749, 0.8358439803123474, 0.8267647624015808, 0.8261701464653015, 0.8259883522987366, 0.825950026512146, 0.8259525895118713, 0.8259526491165161, 0.8260300755500793, 0.8264394402503967, 0.8265833258628845, 0.8268846869468689, 0.8272895216941833, 0.8273523449897766, 0.8274999856948853, 0.8281876444816589, 0.044835563749074936, 0.044823575764894485, 0.044794827699661255, 0.04478639364242554, 0.044822417199611664, 0.04484621062874794, 0.04484988749027252, 0.045124754309654236, 0.046630311757326126, 0.06120123714208603, 0.956801176071167, 0.9599841237068176, 0.9604568481445312, 0.9607641100883484, 0.9609501957893372, 0.9610435366630554, 0.9610957503318787, 0.9611234068870544, 0.961138129234314, 0.961155891418457, 0.9611685276031494, 0.9611769318580627, 0.9611847400665283, 0.9611863493919373, 0.9611804485321045, 0.9611746072769165, 0.9611660838127136, 0.9611521363258362, 0.9611408114433289, 0.9611324071884155, 0.9611231684684753, 0.9611144661903381, 0.9611090421676636, 0.9611009359359741, 0.9610946178436279, 0.9610860347747803, 0.9610751271247864, 0.961061954498291, 0.9610440135002136, 0.9610264897346497, 0.9610077738761902, 0.9609944224357605, 0.9609850645065308, 0.9609768390655518, 0.9609705805778503, 0.9609634280204773, 0.960955023765564, 0.9609442353248596, 0.9609326720237732, 0.9609203934669495, 0.9609106779098511, 0.9609057307243347, 0.9608930945396423, 0.9608355164527893, 0.9607886075973511, 0.9607844352722168, 0.9607823491096497, 0.9607819318771362, 0.9607813358306885, 0.960780918598175, 0.9607805013656616, 0.96077960729599, 0.9607785940170288, 0.9607775211334229, 0.9597811102867126, 0.9594515562057495, 0.9591441750526428, 0.9590252041816711, 0.9589635133743286, 0.9589466452598572, 0.9589080214500427, 0.9588148593902588, 0.9587534070014954, 0.9586744904518127, 0.9590101838111877, 0.9598581194877625, 0.9603170156478882, 0.9603851437568665, 0.9604369401931763, 0.9604737758636475, 0.960506796836853, 0.9605375528335571, 0.9605526328086853, 0.9605668187141418, 0.9605928659439087, 0.9606133699417114, 0.9606264233589172, 0.9606388807296753, 0.9606534838676453, 0.9606642723083496, 0.9606723189353943, 0.9606784582138062, 0.9606841206550598, 0.960693359375, 0.9607136845588684, 0.9607247114181519, 0.9607297778129578, 0.960734486579895, 0.9607396721839905, 0.9607452750205994, 0.9607501029968262, 0.960753858089447, 0.9607572555541992, 0.9607592225074768, 0.9607612490653992, 0.9607627391815186, 0.9607639908790588, 0.9607664942741394, 0.9607694745063782, 0.9607711434364319, 0.960773229598999, 0.9607760906219482, 0.9607784748077393, 0.9607823491096497, 0.960785448551178, 0.9607874155044556, 0.9607902765274048, 0.9607928991317749, 0.9607956409454346, 0.9607988595962524, 0.9608014822006226, 0.9608040452003479, 0.9608060121536255, 0.9608070254325867, 0.9608078598976135, 0.9608091115951538, 0.9608098864555359, 0.9608109593391418, 0.047023482620716095, 0.04777795076370239, 0.04854002967476845, 0.04894048348069191, 0.04970800131559372, 0.06819721311330795, 0.20697271823883057, 0.34594255685806274, 0.5640498399734497, 0.841139018535614, 0.938251793384552, 0.9524689316749573, 0.9561097025871277, 0.9587557911872864, 0.9605550765991211, 0.9610850811004639, 0.9612510204315186, 0.9613202214241028, 0.9613285660743713, 0.9613171219825745, 0.9613109827041626, 0.9613037109375, 0.9612929821014404, 0.9612880349159241, 0.9612832069396973, 0.9612680077552795, 0.9612511396408081, 0.9612415432929993, 0.96123206615448, 0.961224377155304, 0.9612160325050354, 0.9612055420875549, 0.9611961841583252, 0.9611825346946716, 0.9611790180206299, 0.9611820578575134, 0.9611858129501343, 0.961193859577179, 0.9612008333206177, 0.9612088203430176, 0.9612196087837219, 0.9612316489219666, 0.9612451791763306, 0.9612563252449036, 0.9612674713134766, 0.9612727165222168, 0.9612767100334167, 0.9612829685211182, 0.9612874984741211, 0.961292564868927, 0.9612939953804016, 0.9612956643104553, 0.961298406124115, 0.9612998366355896, 0.9613015651702881, 0.9613035917282104, 0.9613037109375, 0.9613031148910522, 0.9613028168678284, 0.9613016843795776, 0.961300253868103, 0.9612981677055359, 0.9612969756126404, 0.9612960815429688, 0.9512988924980164, 0.950250506401062, 0.9502995014190674, 0.9506533741950989, 0.9512262940406799, 0.9588491916656494, 0.960049569606781, 0.9602463841438293, 0.9604498147964478, 0.9605389833450317, 0.9606779217720032, 0.9606868624687195, 0.9607003927230835, 0.9607130289077759, 0.960722804069519, 0.9607308506965637, 0.9607393145561218, 0.9607473611831665, 0.960753858089447, 0.9607595801353455, 0.9607641696929932, 0.9607676267623901, 0.9607719779014587, 0.9607760906219482, 0.96077960729599, 0.9607837796211243, 0.9607876539230347, 0.9607910513877869, 0.9607939124107361, 0.9607968926429749, 0.960800290107727, 0.9608027935028076, 0.9608049392700195, 0.9608067870140076, 0.9608084559440613, 0.9608099460601807, 0.9608114957809448, 0.9608131647109985, 0.9608147144317627, 0.9608156681060791, 0.9608161449432373, 0.9608164429664612, 0.9608161449432373, 0.9608149528503418, 0.9608139395713806, 0.9608134031295776, 0.9608122706413269, 0.9608108401298523, 0.9608089923858643, 0.9608069062232971, 0.9608054757118225, 0.9608041644096375, 0.9608026146888733, 0.9608014822006226, 0.9608007073402405, 0.9608005285263062, 0.960800290107727, 0.9608004093170166, 0.9608010649681091, 0.9608019590377808, 0.9608027935028076, 0.9608040452003479, 0.960805356502533, 0.9608063697814941, 0.9594048261642456, 0.9591091275215149, 0.9589443206787109, 0.9587544798851013, 0.9591087698936462, 0.9593517184257507, 0.9593270421028137, 0.9593111276626587, 0.9592679142951965, 0.9592062830924988, 0.9591566920280457, 0.9591121673583984, 0.9590840935707092, 0.959023118019104, 0.9589323401451111, 0.9587977528572083, 0.9585991501808167, 0.958418607711792, 0.9580991268157959, 0.9577857255935669, 0.9577452540397644, 0.9484772682189941, 0.9244189858436584, 0.8991419076919556, 0.8445878028869629, 0.804407000541687, 0.729913055896759, 0.6808741092681885, 0.6288294792175293, 0.439605176448822, 0.26446110010147095, 0.1951236128807068, 0.9540911316871643, 0.9529505968093872, 0.9587695002555847, 0.9593697190284729, 0.9595539569854736, 0.959775984287262, 0.9599148035049438, 0.9599277377128601, 0.9599435925483704, 0.9599648714065552, 0.9600024819374084, 0.9600593447685242, 0.9601305723190308, 0.9602711200714111, 0.9604085683822632, 0.9604963064193726, 0.9605517387390137, 0.9605911374092102, 0.9605624079704285, 0.960521936416626, 0.9604812860488892, 0.9604281187057495, 0.9604266881942749, 0.9604315757751465, 0.9604247212409973, 0.9604130983352661, 0.9604092240333557, 0.9604116678237915, 0.9604166150093079, 0.9604200124740601, 0.9604206681251526, 0.9604171514511108, 0.9599708318710327, 0.9597830772399902, 0.9596121311187744, 0.9600470662117004, 0.9603290557861328, 0.9603561162948608, 0.960454523563385, 0.9605196118354797, 0.9605166912078857, 0.9605376720428467, 0.9605926871299744, 0.9606543779373169, 0.960696280002594, 0.9607148170471191, 0.9607409834861755, 0.9607855677604675, 0.9608667492866516, 0.9609118700027466, 0.9609361886978149, 0.9609407186508179, 0.9609541296958923, 0.9609730839729309, 0.9609754085540771, 0.960972011089325, 0.960973858833313, 0.9609767198562622, 0.960974395275116, 0.960966169834137, 0.9609594345092773, 0.9609724283218384, 0.9610030651092529, 0.9610294699668884, 0.9610549211502075, 0.9610909223556519, 0.9611227512359619, 0.9611402153968811, 0.9611533284187317, 0.9611619114875793, 0.9611669778823853, 0.9611508846282959, 0.9611048698425293, 0.9611058831214905, 0.9611127972602844, 0.9611160159111023, 0.9611214995384216, 0.961125910282135, 0.9611300826072693, 0.9611377120018005, 0.9611443877220154, 0.9611489176750183, 0.9611536860466003, 0.9611574411392212, 0.9611603617668152, 0.9611647725105286, 0.9611555337905884, 0.9610642790794373, 0.9610706567764282, 0.9610739350318909, 0.9610781073570251, 0.9610837697982788, 0.9610875844955444, 0.9610891342163086, 0.9610904455184937, 0.9610921144485474, 0.046027906239032745, 0.046093933284282684, 0.046151261776685715, 0.04617181047797203, 0.04634292051196098, 0.04988504573702812, 0.08084939420223236, 0.07948970794677734, 0.07927986979484558, 0.08050709217786789, 0.08154850453138351, 0.0817970484495163, 0.08286011964082718, 0.08264830708503723, 0.08139368891716003, 0.08033452928066254, 0.0901334285736084, 0.0971757248044014, 0.12586991488933563, 0.17062750458717346, 0.20743457973003387, 0.2638722062110901, 0.3246619999408722, 0.4093618392944336, 0.5133767127990723, 0.5748510956764221, 0.6187354326248169, 0.6425226926803589, 0.6611733436584473, 0.6747264266014099, 0.683915376663208, 0.6869024634361267, 0.6858908534049988, 0.6884987354278564, 0.6919183731079102, 0.6988677978515625, 0.7071949243545532, 0.7117608189582825, 0.722254753112793, 0.7318404316902161, 0.7375367879867554, 0.7495707273483276, 0.8874486088752747, 0.9329641461372375, 0.9523055553436279, 0.9582244157791138, 0.9592129588127136, 0.9594770669937134, 0.9597330093383789, 0.9601989388465881, 0.960500955581665, 0.9606971740722656, 0.9608425498008728, 0.9610036015510559, 0.9611351490020752, 0.9612870216369629, 0.9613246321678162, 0.9613456726074219, 0.9613585472106934, 0.9613668918609619, 0.9613752961158752, 0.9613839983940125, 0.9613949060440063, 0.9614073634147644, 0.9598928093910217, 0.9598231911659241, 0.9598050713539124, 0.9598312973976135, 0.9599825739860535, 0.9603023529052734, 0.9604724645614624, 0.9605318307876587, 0.9605730772018433, 0.9606035947799683, 0.960634171962738, 0.9606589078903198, 0.9606757164001465, 0.9606872797012329, 0.9607006907463074, 0.9607113599777222, 0.9607172012329102, 0.9607217311859131, 0.9607259035110474, 0.9607312679290771, 0.9607377052307129, 0.9607440829277039, 0.9607516527175903, 0.9607537388801575, 0.9607542753219604, 0.9607637524604797, 0.9607659578323364, 0.9607590436935425, 0.960760235786438, 0.9607645273208618, 0.9607648253440857, 0.9607688188552856, 0.9607757329940796, 0.9607876539230347, 0.9608163833618164, 0.9608407616615295, 0.9608575105667114, 0.9608742594718933, 0.9608898758888245, 0.9609014391899109, 0.9609084725379944, 0.9609122276306152, 0.96091628074646, 0.96091628074646, 0.9609160423278809, 0.9609193801879883, 0.9609206914901733, 0.9609236717224121, 0.9609331488609314, 0.9609426259994507, 0.9609497785568237, 0.9609552621841431, 0.9609628319740295, 0.9609709978103638, 0.9609785676002502, 0.9609843492507935, 0.960990846157074, 0.9609965682029724, 0.9609977602958679, 0.9609989523887634, 0.9610000848770142, 0.961000919342041, 0.9610016942024231, 0.9610024690628052, 0.9555175304412842, 0.9556953310966492, 0.9562618136405945, 0.9569922089576721, 0.9578225612640381, 0.9592887759208679, 0.9597814679145813, 0.9599905014038086, 0.9601374864578247, 0.9602334499359131, 0.9602946639060974, 0.9603384137153625, 0.9603743553161621, 0.9604043960571289, 0.960426926612854, 0.9604465961456299, 0.9604626893997192, 0.9604759812355042, 0.9604858756065369, 0.960495114326477, 0.9605039358139038, 0.9605113863945007, 0.9605184197425842, 0.9605254530906677, 0.9605324864387512, 0.9605404138565063, 0.960547149181366, 0.9605523943901062, 0.96055668592453, 0.9605602025985718, 0.9605650901794434, 0.9605715274810791, 0.9605796933174133, 0.9605892896652222, 0.9605988264083862, 0.9606090784072876, 0.9606180787086487, 0.9606266617774963, 0.9606332778930664, 0.960637092590332, 0.9606398940086365, 0.96064293384552, 0.9606462717056274, 0.960649311542511, 0.9606529474258423, 0.9606560468673706, 0.9606592059135437, 0.9606658220291138, 0.9606792330741882, 0.9606887102127075, 0.960695207118988, 0.9606976509094238, 0.9606991410255432, 0.9607018232345581, 0.9607042074203491, 0.9607052206993103, 0.9607078433036804, 0.9607119560241699, 0.9607126712799072, 0.9607124924659729, 0.9607120156288147, 0.9607113599777222, 0.9607117176055908, 0.9607119560241699, 0.10794202238321304, 0.05375318229198456, 0.053348273038864136, 0.06989960372447968, 0.11227090656757355, 0.13809891045093536, 0.9564633369445801, 0.9602112174034119, 0.9608790874481201, 0.9610496163368225, 0.9611652493476868, 0.9611321687698364, 0.9610380530357361, 0.960959792137146, 0.9610227346420288, 0.960928738117218, 0.9606345891952515, 0.9606188535690308, 0.9606157541275024, 0.9606167674064636, 0.9606183171272278, 0.9606190919876099, 0.9606198668479919, 0.9606221318244934, 0.9606246948242188, 0.9606260061264038, 0.9605770707130432, 0.9605579376220703, 0.9605523347854614, 0.960554301738739, 0.96055668592453, 0.9605569243431091, 0.960541844367981, 0.9603827595710754, 0.9601548314094543, 0.9601567387580872, 0.9601638913154602, 0.960171103477478, 0.9601826667785645, 0.9601998925209045, 0.960219144821167, 0.9602368474006653, 0.9602511525154114, 0.9602691531181335, 0.9603122472763062, 0.9601315855979919, 0.9601486921310425, 0.9601585865020752, 0.9601691365242004, 0.9601766467094421, 0.9601800441741943, 0.9601820111274719, 0.9601843953132629, 0.9601864218711853, 0.9601869583129883, 0.9601867198944092, 0.9601842164993286, 0.9601861834526062, 0.9601899981498718, 0.9601939916610718, 0.9602022171020508, 0.9602087140083313, 0.9602130055427551, 0.9602189064025879, 0.07308033853769302, 0.06472870707511902, 0.06281701475381851, 0.07542651891708374, 0.958489716053009, 0.9597437977790833, 0.9600203633308411, 0.9609366655349731, 0.9607999324798584, 0.9607911705970764, 0.9608004093170166, 0.9607751965522766, 0.9607717990875244, 0.9607805609703064, 0.9607889652252197, 0.9607970714569092, 0.960806131362915, 0.960812509059906, 0.9608194231987, 0.9608203172683716, 0.9608191251754761, 0.9608174562454224, 0.960817813873291, 0.9608144760131836, 0.9608086347579956, 0.9608064293861389, 0.9608148336410522, 0.9608184695243835, 0.9608218669891357, 0.9608274698257446, 0.9608240723609924, 0.9608239531517029, 0.9608261585235596, 0.9608273506164551, 0.9608287811279297, 0.9608309864997864, 0.960832953453064, 0.9608348608016968, 0.9608372449874878, 0.960839569568634, 0.9608418941497803, 0.9608445167541504, 0.9608469605445862, 0.9608492851257324, 0.9608519077301025, 0.9608539938926697, 0.9608559608459473, 0.9608586430549622, 0.9608603715896606, 0.9608619213104248, 0.9608632326126099, 0.9608641266822815, 0.9608646631240845, 0.9608655571937561, 0.9608667492866516, 0.9608678817749023, 0.9608688354492188, 0.9608702659606934, 0.9608721733093262, 0.9608737230300903, 0.9608751535415649, 0.9608765840530396, 0.9608777761459351, 0.9608789682388306, 0.044128213077783585, 0.044001709669828415, 0.04398318752646446, 0.36621490120887756, 0.9604712724685669, 0.960878849029541, 0.9608350396156311, 0.9608485102653503, 0.9608568549156189, 0.9608578681945801, 0.9608572125434875, 0.9608615040779114, 0.9608557820320129, 0.9608513712882996, 0.9608455300331116, 0.960837721824646, 0.9608353972434998, 0.9608331918716431, 0.9608300924301147, 0.9608208537101746, 0.9608073234558105, 0.9607908129692078, 0.9607658386230469, 0.9607574939727783, 0.9607582688331604, 0.9607585668563843, 0.9607580304145813, 0.9607581496238708, 0.9607547521591187, 0.9607528448104858, 0.9607507586479187, 0.9607451558113098, 0.9607396721839905, 0.9607298374176025, 0.9607211947441101, 0.9607175588607788, 0.9607139229774475, 0.9607090950012207, 0.9607075452804565, 0.9607069492340088, 0.960702121257782, 0.9606986045837402, 0.9606962203979492, 0.9606924653053284, 0.960689902305603, 0.960695207118988, 0.9606987237930298, 0.9607053399085999, 0.9607104063034058, 0.9607100486755371, 0.96070796251297, 0.9607030153274536, 0.9607003927230835, 0.9606988430023193, 0.9606966376304626, 0.9606947898864746, 0.9606916904449463, 0.9606901407241821, 0.960689127445221, 0.9606870412826538, 0.9606842994689941, 0.9606819748878479, 0.960681140422821, 0.9606802463531494, 0.06602467596530914, 0.06420862674713135, 0.06276628375053406, 0.11723052710294724, 0.7084721922874451, 0.9590867161750793, 0.959807813167572, 0.960091233253479, 0.9602137804031372, 0.960337221622467, 0.9604299068450928, 0.9604746699333191, 0.960500180721283, 0.9605220556259155, 0.9605368971824646, 0.9605467915534973, 0.960554838180542, 0.9605636596679688, 0.9605883955955505, 0.9606096148490906, 0.9606223702430725, 0.9606379866600037, 0.960655689239502, 0.9606735110282898, 0.9606860876083374, 0.9606911540031433, 0.9606947898864746, 0.9606961011886597, 0.9606970548629761, 0.9607023596763611, 0.9607048034667969, 0.9607119560241699, 0.9607232809066772, 0.9607245922088623, 0.9607218503952026, 0.9607207179069519, 0.960720956325531, 0.9607210755348206, 0.9607208371162415, 0.9607208371162415, 0.9607211947441101, 0.9607219696044922, 0.9607225060462952, 0.9607232809066772, 0.9607236981391907, 0.9607236981391907, 0.9607238173484802, 0.9607243537902832, 0.9607255458831787, 0.960726797580719, 0.9607276916503906, 0.960728645324707, 0.9607295393943787, 0.9607304334640503, 0.9607313871383667, 0.9607319831848145, 0.960732638835907, 0.96073317527771, 0.9607332944869995, 0.9607334733009338, 0.9607340693473816, 0.9607348442077637, 0.960735559463501, 0.9607364535331726, 0.9595941305160522, 0.9594888091087341, 0.9595330357551575, 0.9596540927886963, 0.9597006440162659, 0.9597490429878235, 0.9599541425704956, 0.960141122341156, 0.9602057337760925, 0.9602153301239014, 0.9602695107460022, 0.9603663682937622, 0.9604437351226807, 0.9604890942573547, 0.9605209827423096, 0.9605550765991211, 0.9605838656425476, 0.96060711145401, 0.9606260061264038, 0.9606415033340454, 0.9606584310531616, 0.9606665968894958, 0.9606746435165405, 0.9606817960739136, 0.9606865048408508, 0.9606848359107971, 0.9606792330741882, 0.9606735110282898, 0.9606695771217346, 0.9606581330299377, 0.9606420397758484, 0.9606319665908813, 0.96063631772995, 0.9606392979621887, 0.9606450200080872, 0.9606528282165527, 0.9606505632400513, 0.9606484770774841, 0.9606471061706543, 0.9606464505195618, 0.9606528282165527, 0.9606598615646362, 0.9606683850288391, 0.9606770277023315, 0.9606855511665344, 0.9606961011886597, 0.9607058763504028, 0.9607148766517639, 0.9607229232788086, 0.9607283473014832, 0.9607282280921936, 0.9607279896736145, 0.9607307314872742, 0.960732638835907, 0.9607342481613159, 0.9607374668121338, 0.9607430696487427, 0.9607484340667725, 0.9607535004615784, 0.96075838804245, 0.9607612490653992, 0.9607639908790588, 0.9607680439949036, 0.9607717990875244, 0.9498513340950012, 0.9528881907463074, 0.9565755724906921, 0.9583367109298706, 0.9590715765953064, 0.959519624710083, 0.9593840837478638, 0.960491955280304, 0.9612461924552917, 0.9612473845481873, 0.9612447619438171, 0.9612457156181335, 0.96124267578125, 0.9612376093864441, 0.9612337350845337, 0.9612299799919128, 0.961225688457489, 0.9612202048301697, 0.9612141251564026, 0.9612069725990295, 0.9612008333206177, 0.9611942768096924, 0.961187481880188, 0.9611818790435791, 0.9611761569976807, 0.9611677527427673, 0.9611576199531555, 0.9611507058143616, 0.961143970489502, 0.9611350297927856, 0.9611313343048096, 0.9611300826072693, 0.9611271023750305, 0.961124837398529, 0.9611219763755798, 0.9611185193061829, 0.9611167907714844, 0.9611150026321411, 0.961112380027771, 0.9611107110977173, 0.9611093997955322, 0.9611082077026367, 0.9611077308654785, 0.9611075520515442, 0.9611073136329651, 0.961107075214386, 0.961107075214386, 0.9611068367958069, 0.9611063003540039, 0.9611053466796875, 0.9611032605171204, 0.9611009359359741, 0.9610981941223145, 0.9610959887504578, 0.9610949754714966, 0.961094081401825, 0.9610939621925354, 0.9610937833786011, 0.9610936641693115, 0.961093544960022, 0.961093544960022, 0.9610936641693115, 0.9610938429832458, 0.9610937833786011, 0.04612163081765175, 0.04539564996957779, 0.04513335973024368, 0.044979557394981384, 0.044915031641721725, 0.04498179256916046, 0.045165903866291046, 0.04531339555978775, 0.045249465852975845, 0.04656067118048668, 0.050991468131542206, 0.07386419177055359, 0.2171202152967453, 0.6357107758522034, 0.8641313910484314, 0.9264901876449585, 0.9406841397285461, 0.9467298984527588, 0.950497031211853, 0.9535072445869446, 0.9550473690032959, 0.9557539820671082, 0.956321120262146, 0.956628143787384, 0.9568573832511902, 0.9571123123168945, 0.9573560953140259, 0.9576050639152527, 0.9576463103294373, 0.9570162296295166, 0.9579169154167175, 0.9590357542037964, 0.9589169025421143, 0.9586833715438843, 0.9584958553314209, 0.95835280418396, 0.9582669138908386, 0.9582008719444275, 0.9581194519996643, 0.9580867290496826, 0.9579801559448242, 0.9579275846481323, 0.9577746391296387, 0.9575415849685669, 0.9573606252670288, 0.9572165012359619, 0.9571077227592468, 0.9571932554244995, 0.957353413105011, 0.9575561881065369, 0.9579572081565857, 0.9581583142280579, 0.9583250284194946, 0.9583890438079834, 0.9584923386573792, 0.9584558606147766, 0.9583688378334045, 0.9582651257514954, 0.9581367373466492, 0.9580139517784119, 0.9578753709793091, 0.9576961398124695, 0.9575574994087219, 0.9575848579406738, 0.9578158855438232, 0.9546317458152771, 0.9465726017951965, 0.9500429034233093, 0.9556547999382019, 0.9595007300376892, 0.9600796103477478, 0.9602361917495728, 0.9602974057197571, 0.9603338241577148, 0.9603792428970337, 0.9604169130325317, 0.960440993309021, 0.9604601263999939, 0.9604779481887817, 0.9604952335357666, 0.9605104923248291, 0.9605230689048767, 0.9605351686477661, 0.9605487585067749, 0.960560142993927, 0.9605690240859985, 0.9605734348297119, 0.9605777263641357, 0.9606071710586548, 0.9606361389160156, 0.9606454968452454, 0.960655927658081, 0.9606660604476929, 0.9606735110282898, 0.9606936573982239, 0.9607185125350952, 0.9607366919517517, 0.9607591032981873, 0.9607672691345215, 0.9607738852500916, 0.9607848525047302, 0.960794985294342, 0.9608127474784851, 0.9608195424079895, 0.9608264565467834, 0.9608389139175415, 0.9608476161956787, 0.9608554244041443, 0.9608623385429382, 0.9608688354492188, 0.9608749151229858, 0.9608797430992126, 0.9608845710754395, 0.9608895182609558, 0.9608924984931946, 0.9608952403068542, 0.9608992338180542, 0.960902750492096, 0.9609066247940063, 0.9609091281890869, 0.9609110951423645, 0.9609122276306152, 0.9609139561653137, 0.9609169363975525, 0.9609192609786987, 0.9609208106994629, 0.96092289686203, 0.9609246850013733, 0.045246705412864685, 0.04515482112765312, 0.045071545988321304, 0.0455128476023674, 0.05294632539153099, 0.4340451657772064, 0.9542953372001648, 0.9570691585540771, 0.9587184190750122, 0.9594247937202454, 0.9595716595649719, 0.9596937298774719, 0.959911584854126, 0.9600433111190796, 0.960064172744751, 0.9600729942321777, 0.9600783586502075, 0.960077166557312, 0.9600754380226135, 0.9600724577903748, 0.9600747227668762, 0.9601008892059326, 0.9601042866706848, 0.9601091146469116, 0.960112452507019, 0.9601162672042847, 0.9601214528083801, 0.9601374864578247, 0.9601581692695618, 0.9601637721061707, 0.9601636528968811, 0.9601652026176453, 0.9601672887802124, 0.9601668119430542, 0.9601663947105408, 0.9601665139198303, 0.9601677060127258, 0.9601696729660034, 0.9601746201515198, 0.960162341594696, 0.9601590633392334, 0.9601584672927856, 0.9601581692695618, 0.9601585865020752, 0.9601582884788513, 0.9601597189903259, 0.9601567387580872, 0.9601568579673767, 0.960157036781311, 0.9601575136184692, 0.9601589441299438, 0.960168182849884, 0.9601795673370361, 0.9601821303367615, 0.9601834416389465, 0.9601860642433167, 0.9601879119873047, 0.9601911306381226, 0.9601914286613464, 0.9601911306381226, 0.9601917862892151, 0.9601926803588867, 0.9601951837539673, 0.9601993560791016]

        # with wandb.init(project='BCI'):
        #     for i in range(int(len(left)/64)):
        #         # plt.subplot(1,2,1)
        #         plt.plot(x_axis, left[i*64:(i+1)*64])
        #         plt.title("Left Hand")
        #         plt.xlabel('time(token)')
        #         plt.ylabel('prob')
        #         plt.ylim([0,1])
        #         wandb.log({"Left Hand": wandb.Image(plt)})
        #     plt.show()
        #     for i in range(int(len(right)/64)):
        #         # plt.subplot(1,2,2)
        #         plt.plot(x_axis, right[i * 64:(i + 1) * 64])
        #         plt.title("Right Hand")
        #         plt.xlabel('time(token)')
        #         plt.ylabel('prob')
        #         plt.ylim([0, 1])
        #         wandb.log({"Right Hand": wandb.Image(plt)})
        #
        #     plt.show()

    def data_check(self, subject):
        data = np.load(f"C:/Users/junhee/PycharmProjects/BCI/Lee_dataset/subj_{subject}_data.npy")
        label = np.load(f"C:/Users/junhee/PycharmProjects/BCI/Lee_dataset/subj_{subject}_label.npy")

        print("data is ", np.shape(data), data)
        print("label is ", np.shape(label), label)


if __name__ == '__main__':
    # wandb.login()
    # wandb.init(project="BCI", entity="junheejo")

    for i in range(1, 11):
        Train(dataset='Lee').SVtrain(subject=i, fold=1)
    # Train(dataset='Lee').subject_dependent_category_evaluation(subject=6, fold=1)
    # Train(dataset='Lee').dep_eval(subject=6, fold=1)
    Train(dataset='Lee').inf()

    # Train(dataset='Lee').data_check(6)



    pass