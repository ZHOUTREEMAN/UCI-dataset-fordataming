import pandas as pd
from pandas import DataFrame
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
import random
from sklearn import preprocessing

class MiningInit:

    def __init__(self):
        self.data = self.__read_all()
        self.validate_data = self.__read_validate()
        self.data_list = []

    def __read_validate(self):
        mor_data = pd.read_csv("./mfeat-test-0528/mfeat-mor-right-Unseen", delim_whitespace=True)
        length, mor_width = mor_data.shape
        zer_data = pd.read_csv("./mfeat-test-0528/mfeat-zer-right-Unseen", delim_whitespace=True)
        length, zer_width = zer_data.shape
        pix_data = pd.read_csv("./mfeat-test-0528/mfeat-pix-right-Unseen", delim_whitespace=True)
        length, pix_width = pix_data.shape
        mor_col = ["mor" + str(i) for i in range(mor_width)]
        zer_col = ["zer" + str(i) for i in range(zer_width)]
        pix_col = ["pix" + str(i) for i in range(pix_width)]
        mor_data = pd.read_csv("./mfeat-test-0528/mfeat-mor-right-Unseen", delim_whitespace=True, names=mor_col)
        zer_data = pd.read_csv("./mfeat-test-0528/mfeat-zer-right-Unseen", delim_whitespace=True, names=zer_col)
        pix_data = pd.read_csv("./mfeat-test-0528/mfeat-pix-right-Unseen", delim_whitespace=True, names=pix_col)
        mor_data = mor_data[mor_data.columns[0:4]]
        data = pd.concat([mor_data, zer_data, pix_data], axis=1)
        self.mor_col = mor_col[0:4]
        self.pix_col = pix_col
        self.zer_col = zer_col
        return data

    def __read_all(self):
        mor_data = pd.read_csv("./mfeat/mfeat-mor-left", delim_whitespace=True)
        length, mor_width = mor_data.shape
        zer_data = pd.read_csv("./mfeat/mfeat-zer-left", delim_whitespace=True)
        length, zer_width = zer_data.shape
        pix_data = pd.read_csv("./mfeat/mfeat-pix-left", delim_whitespace=True)
        length, pix_width = pix_data.shape
        mor_col = ["mor" + str(i) for i in range(mor_width)]
        zer_col = ["zer" + str(i) for i in range(zer_width)]
        pix_col = ["pix" + str(i) for i in range(pix_width)]
        mor_data = pd.read_csv("./mfeat/mfeat-mor-left", delim_whitespace=True, names=mor_col)
        zer_data = pd.read_csv("./mfeat/mfeat-zer-left", delim_whitespace=True, names=zer_col)
        pix_data = pd.read_csv("./mfeat/mfeat-pix-left", delim_whitespace=True, names=pix_col)
        mor_data = mor_data[mor_data.columns[0:4]]
        data = pd.concat([mor_data, zer_data, pix_data], axis=1)
        self.mor_col = mor_col[0:4]
        self.pix_col = pix_col
        self.zer_col = zer_col
        data = self.__add_label(data, 160)
        return data


    def __add_label(self, data, num):
        y = []
        for i in range(0, 10):
            for j in range(num):
                y.append(i)
        res = DataFrame(y)
        res.columns = ["label"]
        data = pd.concat([data, res], axis=1)
        return data


    def __divide_by_label(self, data, start, end):
        res_data = DataFrame()
        for i in range(start, end + 1):
            temp_data = data[data["label"] == i]
            length, width = temp_data.shape
            num_list = random.sample(range(0, length), 16)
            for index in num_list:
                row = temp_data.iloc[index, :]
                row = DataFrame(row).T
                res_data = pd.concat([res_data, row])
                row_index = row.index.values[0]
                data.drop(row_index, inplace=True)
        return res_data, data


    def __ten_cross_validate(self, vali_num):
        for i in range(vali_num):
            res, self.data = self.__divide_by_label(self.data, 0, 9)
            self.data.reset_index()
            self.data_list.append(res)


    def fit_transform(self):
        self.__ten_cross_validate(10)
        self.__processing()

    def __deal(self, train_data, test_data, label, algo):
        x = np.array(train_data)
        y = np.array(label)
        algo.fit(x, y)
        test_x = np.array(test_data)
        res_x = algo.transform(x)
        res_test = algo.transform(test_x)
        res_x = DataFrame(res_x)
        res_test = DataFrame(res_test)
        res_x.columns = train_data.columns[0:9]
        res_test.columns = test_data.columns[0:9]
        return res_x, res_test

    def fit_validate(self):
        print(self.validate_data)
        mor_data = self.validate_data[self.mor_col]
        mor_data = self.min_max.transform(np.array(mor_data))
        mor_data = DataFrame(mor_data)
        mor_data.columns = self.mor_col[0:4]
        zer_data = self.validate_data[self.zer_col]
        zer_data = self.zer_lda.transform(np.array(zer_data))
        zer_data = DataFrame(zer_data)
        zer_data.columns = self.zer_col[0:9]
        pix_data = self.validate_data[self.pix_col]
        pix_data = self.pix_lda.transform(np.array(pix_data))
        pix_data = DataFrame(pix_data)
        pix_data.columns = self.pix_col[0:9]
        data = pd.concat([mor_data, zer_data, pix_data], axis=1)
        data.to_csv("cross/validate.csv", index=None)
        return data



    def __processing(self):
        test_list = random.sample(range(0, 10), 10)
        for i in range(10):
            train_data = DataFrame()
            test_data = DataFrame()
            for j in range(10):
                if j == test_list[i]:
                    test_data = pd.concat([test_data, self.data_list[j]])
                else:
                    train_data = pd.concat([train_data, self.data_list[j]])
            train_data = train_data.reset_index()
            test_data = test_data.reset_index()
            self.zer_lda = LinearDiscriminantAnalysis(n_components=10)
            zer_train, zer_test = self.__deal(train_data[self.zer_col], test_data[self.zer_col], train_data["label"], self.zer_lda)
            self.pix_lda = LinearDiscriminantAnalysis(n_components=10)
            pix_train, pix_test = self.__deal(train_data[self.pix_col], test_data[self.pix_col], train_data["label"], self.pix_lda)
            mor_train = np.array(train_data[self.mor_col])
            mor_test = np.array(test_data[self.mor_col])
            min_max = preprocessing.MinMaxScaler()
            mor_train = DataFrame(min_max.fit_transform(mor_train))
            mor_test = DataFrame(min_max.transform(mor_test))
            self.min_max = min_max
            mor_train.columns = self.mor_col
            mor_test.columns = self.mor_col
            train_data = pd.concat([mor_train, zer_train, pix_train, train_data["label"]], axis=1)
            test_data = pd.concat([mor_test, zer_test, pix_test, test_data["label"]], axis=1)
            train_data.to_csv("./cross/train" + str(i) + ".csv", index=None)
            test_data.to_csv("./cross/test" + str(i) + ".csv", index=None)
            print(train_data.shape)
            print(test_data.shape)

if __name__ == "__main__":
    init = MiningInit()
    init.fit_transform()
    init.fit_validate()