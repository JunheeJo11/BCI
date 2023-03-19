
import numpy as np
from moabb.datasets import BNCI2014001, PhysionetMI, Cho2017, Lee2019_MI
from moabb.paradigms import MotorImagery
import scipy.io
from scipy.signal import resample


class BCIC:
    def __init__(self):
        self._dataset = BNCI2014001()
        self._paradigm = MotorImagery(n_classes=4)
        self._subjects = list(np.linspace(1, 9, 9, dtype=int))
        self._parent_dir = 'BCIC_dataset'

    def get_dataset(self):
        for subject in self._subjects:
            self.get_subject_data(subject)

    def get_subject_data(self, subject):
        data, label, meta = self._paradigm.get_data(dataset=self._dataset, subjects=[subject])
        _, label = np.unique(label, return_inverse=True)
        for i in range(data.shape[0]):
            target_mean = np.mean(data[i])
            target_std = np.std(data[i])
            data[i] = (data[i] - target_mean) / target_std
        np.save(f'{self._parent_dir}/subj_{subject}_data', data.transpose((0, 2, 1)))
        np.save(f'{self._parent_dir}/subj_{subject}_label', label[:, np.newaxis])


class PhysioNet:
    def __init__(self):
        self._dataset = PhysionetMI()
        self._paradigm = MotorImagery(n_classes=4, tmin=0, tmax=4)
        self._subjects = list(np.linspace(1, 109, 109, dtype=int))
        self._subjects.remove(87)
        self._subjects.remove(90)
        self._subjects.remove(97)

        self._parent_dir = 'PhysioNet_dataset'

    def get_dataset(self):
        for subject in self._subjects:
            self.get_subject_data(subject)

    def get_subject_data(self, subject):
        data, label, meta = self._paradigm.get_data(dataset=self._dataset, subjects=[subject])
        _, label = np.unique(label, return_inverse=True)
        for i in range(data.shape[0]):
            target_mean = np.mean(data[i])
            target_std = np.std(data[i])
            data[i] = (data[i] - target_mean) / target_std
        np.save(f'{self._parent_dir}/subj_{subject}_data', data.transpose((0,2,1)))
        np.save(f'{self._parent_dir}/subj_{subject}_label', label[:, np.newaxis])


class Cho:
    def __init__(self):
        self._dataset = Cho2017()
        self._paradigm = MotorImagery(n_classes=2, tmin=0, tmax=4)
        self._subjects = list(np.linspace(1, 52, 52, dtype=int))
        self._subjects.remove(32)
        self._subjects.remove(46)
        self._subjects.remove(49)

        self._parent_dir = 'Cho_dataset'

    def get_dataset(self):
        for subject in self._subjects:
            self.get_subject_data(subject)

    def get_subject_data(self, subject):
        data, label, meta = self._paradigm.get_data(dataset=self._dataset, subjects=[subject])
        _, label = np.unique(label, return_inverse=True)
        for i in range(data.shape[0]):
            target_mean = np.mean(data[i])
            target_std = np.std(data[i])
            data[i] = (data[i] - target_mean) / target_std

        np.save(f'{self._parent_dir}/subj_{subject}_data', data.transpose((0,2,1)))
        np.save(f'{self._parent_dir}/subj_{subject}_label', label[:, np.newaxis])


class LeeMI:
    def __init__(self):
        self._dataset = Lee2019_MI(test_run=False)
        self._subjects = list(np.linspace(1, 54, 54, dtype=int))
        self._parent_dir = 'Lee_dataset'

    def get_dataset(self):
        for subject in self._subjects:
            self._dataset.download(subject_list=[subject])
            self.get_subject_data(subject)

    def get_subject_data(self, subject):
        self.access_subject_file(subject)
        np.save(f'{self._parent_dir}/subj_{subject}_data', self._data)
        np.save(f'{self._parent_dir}/subj_{subject}_label', self._label)

    def access_subject_file(self, subject):
        mat_dir = self._dataset.data_path(subject)
        ses1_data = scipy.io.loadmat(mat_dir[0])
        ses1_data, ses1_label = self.load_session_data(ses1_data)
        ses1_data = resample(ses1_data, 1000, axis=1)

        ses2_data = scipy.io.loadmat(mat_dir[1])
        ses2_data, ses2_label = self.load_session_data(ses2_data)
        ses2_data = resample(ses2_data, 1000, axis=1)

        self._data = np.concatenate((ses1_data, ses2_data))
        self._label = np.concatenate((ses1_label, ses2_label))-1

    def load_session_data(self, data):
        train_dataset = data['EEG_MI_train']
        test_dataset = data['EEG_MI_test']

        train_data = train_dataset['smt'].item()
        train_label = train_dataset['y_dec'].item()

        test_data = test_dataset['smt'].item()
        test_label = test_dataset['y_dec'].item()

        data = np.concatenate((train_data, test_data), axis=1)
        label = np.concatenate((train_label, test_label), axis=1)
        return data.transpose((1,0,2)), label.transpose((1,0))


class Dataset:
    def __init__(self, dataset):
        if dataset == 'BCIC':
            self.paradigm = BCIC()
        elif dataset == 'PhysioNet':
            self.paradigm = PhysioNet()
        elif dataset == 'Cho':
            self.paradigm = Cho()
        elif dataset == 'Lee':
            self.paradigm = LeeMI()
        else:
            raise "Dataset is INVALID"

    def get_dataset(self):
        self.paradigm.get_dataset()


if __name__ == '__main__':
    Dataset(dataset='Lee').get_dataset() # key: BCIC, PhysioNet, Cho, lee