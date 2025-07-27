import os
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np


class AudioDataset(Dataset):
    def __init__(self, rootDir, patientList, excelPath, AHILimit, isTrain, repeat_num):
        self.audioPaths_lie = []
        self.audioPaths_sit = []
        # 1=坐位拼音；2=坐位汉字；3=坐位句子；4=仰卧位拼音；5=仰卧位汉字；6=仰卧位句子
        self.audioPaths_lie += [
            os.path.join(rootDir, str(p) + "_3.wav") for p in patientList
        ]
        self.audioPaths_sit += [
            os.path.join(rootDir, str(p) + "_6.wav") for p in patientList
        ]
        self.LbData = pd.read_excel(excelPath)
        self.AHI = [
            self.LbData[self.LbData.patient == p].AHI.to_list()[0] for p in patientList
        ]
        self.AHILimit = AHILimit
        self.isTrain = isTrain
        self.speaker = list(range(len(patientList)))
        self.patients = patientList

        if self.isTrain:
            self.audioPaths_lie = repeat_num * self.audioPaths_lie
            self.audioPaths_sit = repeat_num * self.audioPaths_sit
            self.AHI = repeat_num * self.AHI
            self.speaker = repeat_num * self.speaker
            self.patients = repeat_num * self.patients

    def __getitem__(self, item):
        # data, sampleRate = soundfile.read(self.audioPaths[item])
        lb = (self.AHI[item] > self.AHILimit) * 1
        spk = self.speaker[item]
        patienttt = self.patients[item]
        return self.audioPaths_lie[item], self.audioPaths_sit[item], lb, spk, patienttt

    def __len__(self):
        return len(self.audioPaths_lie)
