# this is inherited from siamese_net_v3_xlsr.py. what i plan to do are:
# add one more index: f1 score
# calculate accuracy on full training set at the end
# freeze encoder during test epoch, unfreeze afterwards

# current version: only fold 5, fix seed

from data_full import AudioDataset
from audioModel_shared_new import modelForSequenceClassification

from torch.utils.data import DataLoader
import torchaudio
import pandas as pd
from tqdm import tqdm

# import soundfile
import os
import shutil
import random
import numpy as np

# from datasets import load_dataset
from transformers import Wav2Vec2FeatureExtractor
import argparse
from time import strftime
from time import localtime
import logging

# from pytorch_revgrad import RevGrad

import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import LambdaLR
import torch

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import KFold

parser = argparse.ArgumentParser(description="osa")
parser.add_argument("--n_epochs", type=int, default=50)
parser.add_argument(
    "--test_epochs", type=int, default=10
)  # after which we don't check model failure
parser.add_argument(
    "--checkpoint_epochs", type=int, default=100
)  # how frequence we check model failure before tolerance_epochs
parser.add_argument(
    "--tolerance_epochs", type=int, default=100
)  # after which we don't allow model failure
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--update_interval", type=int, default=2)
parser.add_argument("--test_batch_size", type=int, default=1)
# parser.add_argument('--cut_length', type=int, default=24000)
parser.add_argument("--maxLength", type=int, default=96000)
parser.add_argument("--moveLength", type=int, default=16000)

# the code regarding this parameter is wrong: you need to the related part into audioModel.py/audioModel_shared.py
parser.add_argument("--ratio", type=float, default=0.5)

parser.add_argument("--gpu_device", type=int, default=1)
parser.add_argument(
    "--wav2vec2_type", type=str, default="encoder"
)  # none: OSA; encoder: OSA + w2v2 encoder; all: OSA + w2v2 all
parser.add_argument("--log_path", type=str, default="./log_osa/trial")
parser.add_argument("--fig_path", type=str, default="./fig_osa")
parser.add_argument("--acc_path", type=str, default="./acc_osa")
parser.add_argument("--model_osa_path", type=str, default="./model_osa")
parser.add_argument("--split_mode", type=str, default="random3")
parser.add_argument("--num_modules", type=int, default=24)

parser.add_argument("--is_copy_py", action="store_true")
parser.add_argument("--is_save_model", action="store_true")

# parser.add_argument('--balance_train', action='store_false')
# parser.add_argument('--balance_test', action='store_true')
parser.add_argument("--is_relu", action="store_false")
parser.add_argument("--is_adversarial", action="store_true")

parser.add_argument("--alpha_parameter", type=float, default=0.0)
parser.add_argument(
    "--waiting_epochs", type=int, default=0
)  # after which we began training speaker

parser.add_argument(
    "--seed_number", type=int, default=0
)  # this does not affect the generator seed number
parser.add_argument("--is_fixed_generator", action="store_false")

parser.add_argument("--voting_threshold", type=float, default=0.5)

parser.add_argument("--model_choice", type=str, default="wav2vec2_base")

parser.add_argument("--AHILimit", type=int, default=30)

parser.add_argument("--repeat_num", type=int, default=16)

parser.add_argument("--is_encoder_rand", action="store_true")

parser.add_argument("--save_model_epoch", type=int, default=200)

flags = parser.parse_args()

now = strftime("%Y-%m-%d_%H:%M:%S", localtime())


def setup_seed(seed_num):
    torch.manual_seed(seed_num)
    random.seed(seed_num)
    np.random.seed(seed_num)
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed_num)
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)


def print_logging(strr):
    print(strr)
    logging.info(strr)


log_dir = flags.log_path
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

fig_dir = flags.fig_path
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

acc_dir = flags.acc_path
if not os.path.exists(acc_dir):
    os.makedirs(acc_dir)

model_dir = flags.model_osa_path
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

script_name = os.path.basename(__file__)
if flags.is_copy_py:
    shutil.copyfile(script_name, os.path.join(flags.log_path, script_name))

# random_list = ['fold1', 'fold2', 'fold3', 'fold4', 'fold5']
old_epoch = 0
max_f1_epoch = 0
max_f1 = 0
max_ac = 0

# flags.split_mode = 'fold5'

seed_list = list(range(20))

if True:

    # for flags.seed_number in seed_list:

    # for flags.split_mode in random_list:

    setup_seed(flags.seed_number)

    # logging.basicConfig(filename=os.path.join(log_dir, 'log_' + '%s_%d_%d_'
    #                                           % (flags.wav2vec2_type, flags.num_modules, flags.n_epochs) + script_name + '_' + now + '.txt'), level=logging.INFO)
    # logging.basicConfig(filename=os.path.join(log_dir, 'log_' + flags.split_mode + '_' + '%d_%f_%d_%d_'
    #                                           % (flags.num_modules, flags.alpha_parameter, flags.waiting_epochs,
    #                                              flags.maxLength) + 'ad_' + '%d_' % flags.is_adversarial
    #                                           + script_name + '_' + now + '.txt'), level=logging.INFO)
    logging.basicConfig(
        filename=os.path.join(log_dir, "log" + "_" + now + ".txt"), level=logging.INFO
    )

    print_logging("now is:")
    print_logging(now)

    print_logging(script_name)
    print_logging("Flags:")
    for k, v in sorted(vars(flags).items()):
        print_logging("\t{}: {}".format(k, v))
    print_logging("---------------------------------------------------------")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(flags.gpu_device)

    # if you choose not to normalize, you can first normalize the full audio seq before inserting it into processor_f
    # feature_extractor_f = Wav2Vec2FeatureExtractor(do_normalize=False, return_attention_mask=True)
    # tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
    #     os.path.join("/home/zhangkb/.cache/huggingface", flags.model_choice),
    #     local_files_only=True)
    # processor_f = Wav2Vec2Processor(feature_extractor=feature_extractor_f, tokenizer=tokenizer)

    feature_extractor_f = Wav2Vec2FeatureExtractor(
        do_normalize=False, return_attention_mask=True
    )

    # split_list = pd.read_excel('split.xlsx')
    # if flags.split_mode == 'fold1':
    #     test_patientList = list(split_list.random1_high)[:15] + list(split_list.random1_low)[:15]
    #     random.shuffle(test_patientList)
    # if flags.split_mode == 'fold2':
    #     test_patientList = list(split_list.random2_high)[:15] + list(split_list.random2_low)[:15]
    #     random.shuffle(test_patientList)
    # if flags.split_mode == 'fold3':
    #     test_patientList = list(split_list.random3_high)[:15] + list(split_list.random3_low)[:15]
    #     random.shuffle(test_patientList)
    # if flags.split_mode == 'fold4':
    #     test_patientList = list(split_list.random4_high)[:15] + list(split_list.random4_low)[:15]
    #     random.shuffle(test_patientList)
    # if flags.split_mode == 'fold5':
    #     test_patientList = list(split_list.random5_high)[:15] + list(split_list.random5_low)[:15]
    #     random.shuffle(test_patientList)

    dt = pd.read_excel("/ssd/zhangkb/osa_nn/new_data/label_new_data.xlsx")
    patients = list(dt.patient)
    patientList = patients.copy()

    random.shuffle(patientList)

    # 计算每个子列表的大小
    n = len(patientList) // 5
    remainder = len(patientList) % 5

    # 将列表等分成5个部分
    lists = []
    start_idx = 0
    for i in range(5):
        # 如果有余数，分配一个额外的元素给前几个子列表
        end_idx = start_idx + n + (1 if i < remainder else 0)
        lists.append(patientList[start_idx:end_idx])
        start_idx = end_idx

    # train_patientList = [i for i in patientList if i not in test_patientList]
    # speaker_num = len(patientList)

    # reason why codes below are wrong: https://stackoverflow.com/questions/497426/deleting-multiple-elements-from-a-list
    # (del from the end to the beginning will work)
    # for ct in pickup_index:
    #     del patientList[ct]

    # dataSet = AudioDataset(rootDir='/ssd/zhangkb/osa_nn/new_data/audios',
    #                         patientList=patientList.copy(),
    #                         excelPath='/ssd/zhangkb/osa_nn/new_data/label_new_data.xlsx',
    #                         AHILimit=flags.AHILimit,
    #                         isTrain=True,
    #                         repeat_num=flags.repeat_num)

    # trainLoader = DataLoader(dataset=trainSet, batch_size=flags.batch_size, shuffle=True, pin_memory=False)
    # testLoader = DataLoader(dataset=testSet, batch_size=flags.test_batch_size, shuffle=False, pin_memory=False)

    # kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for fold in range(5):

        print_logging(f"Fold {fold + 1}/{5}")

        test_patientList = lists[fold]
        train_patientList = [i for i in patientList if i not in test_patientList]

        print_logging(test_patientList)

        trainSet = AudioDataset(
            rootDir="/ssd/zhangkb/osa_nn/new_data/audios",
            patientList=train_patientList.copy(),
            excelPath="/ssd/zhangkb/osa_nn/new_data/label_new_data.xlsx",
            AHILimit=flags.AHILimit,
            isTrain=True,
            repeat_num=flags.repeat_num,
        )

        testSet = AudioDataset(
            rootDir="/ssd/zhangkb/osa_nn/new_data/audios",
            patientList=test_patientList.copy(),
            excelPath="/ssd/zhangkb/osa_nn/new_data/label_new_data.xlsx",
            AHILimit=flags.AHILimit,
            isTrain=False,
            repeat_num=flags.repeat_num,
        )

        trainLoader = DataLoader(
            dataset=trainSet,
            batch_size=flags.batch_size,
            shuffle=True,
            pin_memory=False,
        )
        testLoader = DataLoader(
            dataset=testSet,
            batch_size=flags.test_batch_size,
            shuffle=False,
            pin_memory=False,
        )

        model_init = modelForSequenceClassification.from_pretrained(
            os.path.join("/home/zhangkb/.cache/huggingface", flags.model_choice),
            local_files_only=True,
        )
        configuration = model_init.config
        configuration.num_hidden_layers = flags.num_modules
        if flags.is_encoder_rand:
            model_shared = modelForSequenceClassification(config=configuration)
        else:
            model_shared = modelForSequenceClassification.from_pretrained(
                os.path.join("/home/zhangkb/.cache/huggingface", flags.model_choice),
                local_files_only=True,
                config=configuration,
            )

        # if flags.is_pretrained:
        #     osa_layer = torch.nn.Linear(256, 2)
        #
        #
        #     class net_osa(nn.Module):
        #         def __init__(self):
        #             super(net_osa, self).__init__()
        #             self.backbone = model_shared
        #             self.osa_layer = osa_layer
        #
        #         def forward(
        #                 self,
        #                 input_values,
        #                 attention_mask=None,
        #                 output_attentions=None,
        #                 output_hidden_states=None,
        #                 return_dict=None,
        #                 labels=None,
        #                 is_relu=True
        #         ):
        #             intermidiate_OSA = self.backbone(input_values,
        #                                              attention_mask=attention_mask,
        #                                              is_relu=is_relu)
        #             logits_OSA = self.osa_layer(intermidiate_OSA)
        #             return logits_OSA
        #
        #
        #     model_osa = net_osa()
        #     if flags.AHILimit == 30:
        #         model_osa.load_state_dict(torch.load(
        #             '/home/zhangkb/OSA3rdPeriod/model_osa/good_models/model_osa_random56_0.100000_20_96000_ad_1_ac_0.745960_epoch_15_audioTrain_v9_nfold.py_2022-02-25_15_43_26.pt'))
        #     if flags.AHILimit == 10:
        #         model_osa.load_state_dict(torch.load(
        #             '/home/zhangkb/OSA3rdPeriod/model_osa/good_models/model_osa_6_0.100000_20_96000_ad_0_ac_0.833333_epoch_28_audioTrain_v9.py_2022-03-05_11_38_53.pt'))
        #     model_shared = model_osa.backbone
        #
        # else:
        #     model_shared = modelForSequenceClassification.from_pretrained(
        #         os.path.join("/home/zhangkb/.cache/huggingface", flags.model_choice),
        #         local_files_only=True, config=configuration)

        model_shared = model_shared.cuda()

        # osa_layer = torch.nn.Sequential(
        #     nn.Linear(512, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 16),
        #     nn.ReLU(),
        #     nn.Linear(16, 2)
        # )
        osa_layer = torch.nn.Sequential(
            nn.Linear(1024, 64), nn.ReLU(), nn.Linear(64, 2)
        )
        osa_layer = osa_layer.cuda()

        # '''try model saving'''
        # ac=0.3
        # PATH = os.path.join(model_dir, 'model_osa_' + '%d_%f_%d_%d_'
        #                     % (flags.num_modules, flags.alpha_parameter, flags.waiting_epochs,
        #                        flags.maxLength) + 'ad_' + '%d_' % flags.is_adversarial
        #                     + 'ac_' + '%f_' % ac + 'epoch_' + '%d_' % epoch
        #                     + script_name + '_' + now + '.pt')
        # torch.save(model_osa.state_dict(), PATH)

        softmax = nn.Softmax(dim=1)
        loss_fct = CrossEntropyLoss()

        ac_osa_train = np.zeros(flags.n_epochs)
        f1_osa_train = np.zeros(flags.n_epochs)
        loss_osa_train = np.zeros(flags.n_epochs)

        ac_osa_test = np.zeros(flags.n_epochs)
        f1_osa_test = np.zeros(flags.n_epochs)
        precision_osa_test = np.zeros(flags.n_epochs)
        sensitivity_osa_test = np.zeros(flags.n_epochs)
        specificity_osa_test = np.zeros(flags.n_epochs)
        loss_osa_test = np.zeros(flags.n_epochs)
        ac_osa_test_unvoted = np.zeros(flags.n_epochs)
        ac_speaker_train = np.zeros(flags.n_epochs)

        wrong_patient_train = []
        wrong_patient_test = []

        lr = 3e-5

        optimizer_osa = torch.optim.AdamW(osa_layer.parameters(), lr=lr)
        optimizer_shared = None
        if True:
            if flags.wav2vec2_type == "none":
                pass
            if flags.wav2vec2_type == "encoder":
                optimizer_shared = torch.optim.AdamW(
                    model_shared.wav2vec2.encoder.parameters(), lr=lr
                )
            if flags.wav2vec2_type == "all":
                optimizer_shared = torch.optim.AdamW(model_shared.parameters(), lr=lr)

        num_warmup_steps = 3000 / (flags.batch_size / 4) * (flags.repeat_num / 16)
        num_constant_steps = 12000 / (flags.batch_size / 4) * (flags.repeat_num / 16)
        num_training_steps = 30000 / (flags.batch_size / 4) * (flags.repeat_num / 16)

        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            elif (
                current_step < num_warmup_steps + num_constant_steps
                and current_step >= num_warmup_steps
            ):
                return float(1)
            return max(
                0.0,
                float(num_training_steps - current_step)
                / float(
                    max(1, num_training_steps - num_warmup_steps - num_constant_steps)
                ),
            )

        scheduler_osa = LambdaLR(optimizer_osa, lr_lambda)
        if optimizer_shared is not None:
            scheduler_shared = LambdaLR(optimizer_shared, lr_lambda)

        for epoch in range(flags.n_epochs):
            # torch.manual_seed(0)
            # random.seed(0)
            # np.random.seed(0)
            # torch.backends.cudnn.benchmark = False

            print_logging("epoch: " + str(epoch))
            print_logging("")

            if True:

                print_logging("training...")

                p_list = []
                t_list = []

                confMatrix_osa = np.zeros((2, 2))
                # confMatrix_speaker = np.zeros((speaker_num, speaker_num))

                model_shared.train()
                osa_layer.train()
                # speaker_layer.train()
                # model_speaker.train()

                loss_count = []
                counter = 0

                # for counter, data in tqdm(enumerate(trainLoader)):
                for data in tqdm(trainLoader):
                    # audioPaths_lie, audioPaths_sit, lbs, spks = data[0], data[1], data[2], data[3], data[4]
                    [audioPaths_lie, audioPaths_sit, lbs, spk, patienttt] = data
                    # audios = []  # first lie then sit
                    audios_lie = []
                    audios_sit = []
                    for (ap_lie, ap_sit) in zip(audioPaths_lie, audioPaths_sit):
                        audio_lie, sr_lie = torchaudio.load(ap_lie)
                        audio_lie = audio_lie[0].numpy()
                        audio_lie = (audio_lie - np.mean(audio_lie)) / np.std(audio_lie)

                        audio_sit, sr_sit = torchaudio.load(ap_sit)
                        audio_sit = audio_sit[0].numpy()
                        audio_sit = (audio_sit - np.mean(audio_sit)) / np.std(audio_sit)

                        start = random.randint(
                            0, min(len(audio_lie), len(audio_sit)) - flags.maxLength
                        )
                        duration = random.randint(
                            int(flags.maxLength - sr_lie * 0.4), flags.maxLength
                        )

                        audios_lie.append(audio_lie[start : start + duration])
                        audios_sit.append(audio_sit[start : start + duration])

                    audios = audios_lie + audios_sit

                    feature_all = feature_extractor_f(
                        raw_speech=audios,
                        sampling_rate=16000,
                        return_tensors="pt",
                        max_length=flags.maxLength,
                        padding="max_length",
                    )

                    """train osa"""
                    audio_feature_all = model_shared(
                        feature_all.input_values.float().cuda(),
                        attention_mask=feature_all.attention_mask.cuda(),
                        is_relu=flags.is_relu,
                    )

                    audio_feature_lie = audio_feature_all[: len(audioPaths_lie)]
                    audio_feature_sit = audio_feature_all[len(audioPaths_lie) :]
                    audio_feature = torch.concat(
                        (audio_feature_lie, audio_feature_sit), 1
                    )
                    logits_osa = osa_layer(audio_feature)

                    labels = lbs.cuda()
                    loss_osa = loss_fct(logits_osa.view(-1, 2), labels.view(-1))
                    loss_osa.backward()
                    loss_count.append(loss_osa.detach().clone().cpu())

                    counter = counter + 1
                    if counter % flags.update_interval == 0:
                        counter = 0
                        if optimizer_shared is not None and epoch >= flags.test_epochs:
                            optimizer_shared.step()
                        optimizer_osa.step()
                        model_shared.zero_grad()
                        osa_layer.zero_grad()
                        # speaker_layer.zero_grad()
                    if optimizer_shared is not None:
                        scheduler_shared.step()
                    scheduler_osa.step()

                    """see osa training result"""
                    if len(logits_osa.shape) > 2:
                        logits_osa = torch.mean(logits_osa, dim=1)
                    logits_osa = softmax(logits_osa)
                    logits_osa[:, 1] = (
                        logits_osa[:, 1] * (1 - flags.ratio) / flags.ratio
                    )
                    pred_osa = torch.argmax(logits_osa, dim=1)

                    for num, (p, t) in enumerate(zip(pred_osa, lbs)):
                        if p < 2:
                            p_list.append(p.item())
                            t_list.append(t.item())
                            confMatrix_osa[p, t] += 1
                        if p != t:
                            wrong_patient_train.append(patienttt[num])

                confMatrix_osa = confusion_matrix(t_list, p_list)
                print_logging(confMatrix_osa)
                ac = accuracy_score(t_list, p_list)
                ac_osa_train[epoch] = ac
                print_logging("ac_osa = {}".format(ac))

                f1 = f1_score(t_list, p_list)
                f1_osa_train[epoch] = f1
                print_logging("f1_osa = {}".format(f1))

                loss_average = sum(loss_count) / len(loss_count)
                loss_osa_train[epoch] = loss_average
                print_logging("loss_osa = {}".format(loss_average))

                # print_logging(confMatrix_osa)
                # ac = np.trace(confMatrix_osa) / np.sum(confMatrix_osa)
                # ac_osa_train[epoch] = ac
                # print_logging('ac_osa = {}'.format(ac))
                # loss_average = sum(loss_count) / len(loss_count)
                # loss_osa_train[epoch] = loss_average
                # print_logging('loss_osa = {}'.format(loss_average))
                #
                # result_patient = Counter(wrong_patient_train)
                #
                # print_logging(result_patient)
                #
                # wrong_patient_train = []
                # wrong_patient_test = []
                #
                # if epoch >= flags.waiting_epochs and flags.is_adversarial:
                #     ac = np.trace(confMatrix_speaker) / np.sum(confMatrix_speaker)
                #     ac_speaker_train[epoch] = ac
                #     print_logging('ac_speaker = {}'.format(ac))

            """use test method on train"""
            """
            print_logging('testing for train...')
            confMatrix_osa = np.zeros((2, 2))
            confMatrix_osa_unvoted = np.zeros((2, 2))
            p_list = []
            t_list = []

            model_shared.eval()
            osa_layer.eval()
            loss_count = []
            for data in tqdm(trainLoader4test):
                [audioPaths_lie, audioPaths_sit, lbs, spk, patienttt] = data
                preds_osa_cut = []
                logits_osa_cut = []
                for (ap_lie, ap_sit) in zip(audioPaths_lie, audioPaths_sit):
                    # audio, sr = soundfile.read(ap)
                    audio_lie, sr_lie = torchaudio.load(ap_lie)
                    audio_lie = audio_lie[0].numpy()
                    audio_lie = (audio_lie - np.mean(audio_lie)) / np.std(audio_lie)

                    audio_sit, sr_sit = torchaudio.load(ap_sit)
                    audio_sit = audio_sit[0].numpy()
                    audio_sit = (audio_sit - np.mean(audio_sit)) / np.std(audio_sit)

                cut_num = (min(len(audio_lie), len(audio_sit)) - flags.maxLength) // flags.moveLength
                loss_count_cut = []
                for ct in range(cut_num):
                    audio_lie_cut = audio_lie[ct * flags.moveLength: flags.maxLength + ct * flags.moveLength]
                    feature_lie_cut = feature_extractor_f(audio_lie_cut,
                                                          sampling_rate=16000,
                                                          return_tensors='pt',
                                                          max_length=flags.maxLength,
                                                          padding='max_length')
                    audio_feature_lie_cut = model_shared(feature_lie_cut.input_values.float().cuda(),
                                                         attention_mask=feature_lie_cut.attention_mask.cuda(),
                                                         is_relu=flags.is_relu)

                    audio_sit_cut = audio_sit[ct * flags.moveLength: flags.maxLength + ct * flags.moveLength]
                    feature_sit_cut = feature_extractor_f(audio_sit_cut,
                                                          sampling_rate=16000,
                                                          return_tensors='pt',
                                                          max_length=flags.maxLength,
                                                          padding='max_length')
                    audio_feature_sit_cut = model_shared(feature_sit_cut.input_values.float().cuda(),
                                                         attention_mask=feature_sit_cut.attention_mask.cuda(),
                                                         is_relu=flags.is_relu)

                    audio_feature_cut = torch.concat((audio_feature_lie_cut, audio_feature_sit_cut), 1)
                    logit_osa_cut = osa_layer(audio_feature_cut)
                    if len(logit_osa_cut.shape) > 2:
                        logit_osa_cut = torch.mean(logit_osa_cut, dim=1)
                    logit_osa_cut = softmax(logit_osa_cut)
                    logits_osa_cut.append(logit_osa_cut.detach().clone().cpu())
                    # logit_osa_cut[:, 1] = logit_osa_cut[:, 1]*(1-flags.ratio)/flags.ratio
                    label = lbs.cuda()
                    loss_osa_cut = loss_fct(logit_osa_cut.view(-1, 2), label.view(-1))
                    loss_count_cut.append(loss_osa_cut.detach().clone().cpu())
                    pred_osa_cut = torch.argmax(logit_osa_cut, dim=1)
                    preds_osa_cut.append(pred_osa_cut.detach().clone().cpu())
                loss_count.append(sum(loss_count_cut) / len(loss_count_cut))

                # one criteria
                if sum(preds_osa_cut) / len(preds_osa_cut) > flags.voting_threshold:
                    pred_osa = torch.tensor([1])
                else:
                    pred_osa = torch.tensor([0])

                # another criteria
                # pred_osa = torch.tensor([0])
                # for i in range(len(logits_osa_cut)):
                #     if logits_osa_cut[i][0][1] > flags.judgement_threshold:
                #         pred_osa = torch.tensor([1])
                #         break

                for p, t in zip(pred_osa, lbs):
                    if p < 2:
                        p_list.append(p.item())
                        t_list.append(t.item())
                        confMatrix_osa[p, t] += 1
                for p, t in zip(preds_osa_cut, [lbs] * len(preds_osa_cut)):
                    if p < 2:
                        confMatrix_osa_unvoted[p, t] += 1

            confMatrix_osa = confusion_matrix(t_list, p_list)
            print_logging(confMatrix_osa)
            ac = accuracy_score(t_list, p_list)
            ac_osa_train[epoch] = ac
            print_logging('ac_osa = {}'.format(ac))

            f1 = f1_score(t_list, p_list)
            f1_osa_train[epoch] = f1
            print_logging('f1_osa = {}'.format(f1))

            loss_average = sum(loss_count) / len(loss_count)
            loss_osa_train[epoch] = loss_average
            print_logging('loss_osa = {}'.format(loss_average))
            """

            """with a design strategy, the test batch size must be 1"""
            print_logging("testing for test...")
            confMatrix_osa = np.zeros((2, 2))
            confMatrix_osa_unvoted = np.zeros((2, 2))
            p_list = []
            t_list = []

            model_shared.eval()
            osa_layer.eval()
            loss_count = []
            for data in tqdm(testLoader):
                [audioPaths_lie, audioPaths_sit, lbs, spk, patienttt] = data
                preds_osa_cut = []
                logits_osa_cut = []
                for (ap_lie, ap_sit) in zip(audioPaths_lie, audioPaths_sit):
                    # audio, sr = soundfile.read(ap)
                    audio_lie, sr_lie = torchaudio.load(ap_lie)
                    audio_lie = audio_lie[0].numpy()
                    audio_lie = (audio_lie - np.mean(audio_lie)) / np.std(audio_lie)

                    audio_sit, sr_sit = torchaudio.load(ap_sit)
                    audio_sit = audio_sit[0].numpy()
                    audio_sit = (audio_sit - np.mean(audio_sit)) / np.std(audio_sit)

                cut_num = (
                    min(len(audio_lie), len(audio_sit)) - flags.maxLength
                ) // flags.moveLength
                loss_count_cut = []
                for ct in range(cut_num):
                    audio_lie_cut = audio_lie[
                        ct * flags.moveLength : flags.maxLength + ct * flags.moveLength
                    ]
                    feature_lie_cut = feature_extractor_f(
                        audio_lie_cut,
                        sampling_rate=16000,
                        return_tensors="pt",
                        max_length=flags.maxLength,
                        padding="max_length",
                    )
                    audio_feature_lie_cut = model_shared(
                        feature_lie_cut.input_values.float().cuda(),
                        attention_mask=feature_lie_cut.attention_mask.cuda(),
                        is_relu=flags.is_relu,
                    )

                    audio_sit_cut = audio_sit[
                        ct * flags.moveLength : flags.maxLength + ct * flags.moveLength
                    ]
                    feature_sit_cut = feature_extractor_f(
                        audio_sit_cut,
                        sampling_rate=16000,
                        return_tensors="pt",
                        max_length=flags.maxLength,
                        padding="max_length",
                    )
                    audio_feature_sit_cut = model_shared(
                        feature_sit_cut.input_values.float().cuda(),
                        attention_mask=feature_sit_cut.attention_mask.cuda(),
                        is_relu=flags.is_relu,
                    )

                    audio_feature_cut = torch.concat(
                        (audio_feature_lie_cut, audio_feature_sit_cut), 1
                    )
                    logit_osa_cut = osa_layer(audio_feature_cut)
                    if len(logit_osa_cut.shape) > 2:
                        logit_osa_cut = torch.mean(logit_osa_cut, dim=1)
                    logit_osa_cut = softmax(logit_osa_cut)
                    logits_osa_cut.append(logit_osa_cut.detach().clone().cpu())
                    # logit_osa_cut[:, 1] = logit_osa_cut[:, 1]*(1-flags.ratio)/flags.ratio
                    label = lbs.cuda()
                    loss_osa_cut = loss_fct(logit_osa_cut.view(-1, 2), label.view(-1))
                    loss_count_cut.append(loss_osa_cut.detach().clone().cpu())
                    pred_osa_cut = torch.argmax(logit_osa_cut, dim=1)
                    preds_osa_cut.append(pred_osa_cut.detach().clone().cpu())
                loss_count.append(sum(loss_count_cut) / len(loss_count_cut))

                # one criteria
                if sum(preds_osa_cut) / len(preds_osa_cut) > flags.voting_threshold:
                    pred_osa = torch.tensor([1])
                else:
                    pred_osa = torch.tensor([0])

                # another criteria
                # pred_osa = torch.tensor([0])
                # for i in range(len(logits_osa_cut)):
                #     if logits_osa_cut[i][0][1] > flags.judgement_threshold:
                #         pred_osa = torch.tensor([1])
                #         break

                for p, t in zip(pred_osa, lbs):
                    if p < 2:
                        p_list.append(p.item())
                        t_list.append(t.item())
                        confMatrix_osa[p, t] += 1
                for p, t in zip(preds_osa_cut, [lbs] * len(preds_osa_cut)):
                    if p < 2:
                        confMatrix_osa_unvoted[p, t] += 1

            confMatrix_osa = confusion_matrix(t_list, p_list)
            print_logging(confMatrix_osa)
            ac = accuracy_score(t_list, p_list)
            ac_osa_test[epoch] = ac
            print_logging("ac_osa = {}".format(ac))

            f1 = f1_score(t_list, p_list)
            f1_osa_test[epoch] = f1
            print_logging("f1_osa = {}".format(f1))

            loss_average = sum(loss_count) / len(loss_count)
            loss_osa_test[epoch] = loss_average
            print_logging("loss_osa = {}".format(loss_average))

            # PATH = os.path.join(model_dir, 'model_osa_' + flags.split_mode + '_' + '%d_%f_%d_%d_'
            #                     % (flags.num_modules, flags.alpha_parameter, flags.waiting_epochs,
            #                        flags.maxLength) + 'ad_' + '%d_' % flags.is_adversarial
            #                     + 'ac_' + '%f_' % ac + 'epoch_' + '%d_' % epoch
            #                     + script_name + '_' + now + '.pt')
            # if ac >= 0.83:
            #     torch.save(model_osa.state_dict(), PATH)

            if flags.is_save_model:

                PATH_shared = os.path.join(
                    model_dir,
                    "model_shared_"
                    + str(fold)
                    + "_"
                    + flags.wav2vec2_type
                    + "_"
                    + str(flags.seed_number)
                    + "_"
                    + "f1_"
                    + "%f_" % f1
                    + "epoch_"
                    + "%d_" % epoch
                    + now
                    + ".pt",
                )

                if f1 >= 0.8 or epoch >= flags.save_model_epoch:
                    torch.save(model_shared.state_dict(), PATH_shared)

                PATH_osa = os.path.join(
                    model_dir,
                    "osa_layer_"
                    + str(fold)
                    + "_"
                    + flags.wav2vec2_type
                    + "_"
                    + str(flags.seed_number)
                    + "_"
                    + "f1_"
                    + "%f_" % f1
                    + "epoch_"
                    + "%d_" % epoch
                    + now
                    + ".pt",
                )

                if f1 >= 0.8 or epoch >= flags.save_model_epoch:
                    torch.save(osa_layer.state_dict(), PATH_osa)

        if flags.is_save_model:

            PATH_shared = os.path.join(
                model_dir,
                "model_shared_"
                + str(fold)
                + "_"
                + flags.wav2vec2_type
                + "_"
                + str(flags.seed_number)
                + "_"
                + now
                + ".pt",
            )

            torch.save(model_shared.state_dict(), PATH_shared)

            PATH_osa = os.path.join(
                model_dir,
                "osa_layer_"
                + str(fold)
                + "_"
                + flags.wav2vec2_type
                + "_"
                + str(flags.seed_number)
                + "_"
                + now
                + ".pt",
            )

            torch.save(osa_layer.state_dict(), PATH_osa)

        print_logging("---------------------------------------------------------")

        import matplotlib.pyplot as plt

        (l1,) = plt.plot(ac_osa_train)
        (l2,) = plt.plot(ac_osa_test)
        plt.legend(handles=[l1, l2], labels=["osa ac train", "osa ac test"], loc="best")
        plt.ylabel("osa accuracy")
        # plt.savefig(os.path.join(fig_dir, 'osa_' + flags.split_mode + '_' + '%d_%f_%d_%d_'
        #                          % (flags.num_modules, flags.alpha_parameter, flags.waiting_epochs,
        #                             flags.maxLength) + 'ad_' + '%d_' % flags.is_adversarial
        #                          + script_name + '_' + now + '.png'))
        plt.savefig(
            os.path.join(
                fig_dir,
                "osa_ac_"
                + str(fold)
                + "_"
                + flags.wav2vec2_type
                + "_"
                + str(flags.seed_number)
                + "_"
                + now
                + ".png",
            )
        )
        plt.close()

        # l3,=plt.plot(ac_speaker_train)
        # plt.ylabel('speaker accuracy')
        # plt.savefig(os.path.join(fig_dir, 'speaker_' + flags.split_mode + '_' + '%d_%f_%d_%d_'
        #                                           % (flags.num_modules, flags.alpha_parameter, flags.waiting_epochs, flags.maxLength) + 'ad_' + '%d_' % flags.is_adversarial
        #                                           + script_name + '_' + now + '.png'))
        # plt.close()

        (l4,) = plt.plot(loss_osa_train)
        (l5,) = plt.plot(loss_osa_test)
        plt.legend(handles=[l4, l5], labels=["osa train", "osa test"], loc="best")
        plt.ylabel("osa loss")
        # plt.savefig(os.path.join(fig_dir, 'ref_loss_' + flags.split_mode + '_' + '%d_%f_%d_%d_'
        #                          % (flags.num_modules, flags.alpha_parameter, flags.waiting_epochs,
        #                             flags.maxLength) + 'ad_' + '%d_' % flags.is_adversarial
        #                          + script_name + '_' + now + '.png'))
        plt.savefig(
            os.path.join(
                fig_dir,
                "ref_loss_"
                + str(fold)
                + "_"
                + flags.wav2vec2_type
                + "_"
                + str(flags.seed_number)
                + "_"
                + now
                + ".png",
            )
        )
        plt.close()

        (l6,) = plt.plot(f1_osa_train)
        (l7,) = plt.plot(f1_osa_test)
        plt.legend(handles=[l6, l7], labels=["osa f1 train", "osa f1 test"], loc="best")
        plt.ylabel("osa f1 score")
        # plt.savefig(os.path.join(fig_dir, 'osa_' + flags.split_mode + '_' + '%d_%f_%d_%d_'
        #                          % (flags.num_modules, flags.alpha_parameter, flags.waiting_epochs,
        #                             flags.maxLength) + 'ad_' + '%d_' % flags.is_adversarial
        #                          + script_name + '_' + now + '.png'))
        plt.savefig(
            os.path.join(
                fig_dir,
                "osa_f1_"
                + str(fold)
                + "_"
                + flags.wav2vec2_type
                + "_"
                + str(flags.seed_number)
                + "_"
                + now
                + ".png",
            )
        )
        plt.close()

        import pandas as pd

        accuracies = np.concatenate(
            (
                ac_osa_train.reshape(flags.n_epochs, 1),
                ac_osa_test.reshape(flags.n_epochs, 1),
            ),
            axis=1,
        )
        df = pd.DataFrame(accuracies, columns=["ac_osa_train", "ac_osa_test"])
        # filepath = os.path.join(acc_dir, 'acc_' + flags.split_mode + '_' + '%d_%f_%d_%d_'
        #                         % (flags.num_modules, flags.alpha_parameter, flags.waiting_epochs,
        #                            flags.maxLength) + 'ad_' + '%d_' % flags.is_adversarial
        #                         + script_name + '_' + now + '.xlsx')
        filepath = os.path.join(
            acc_dir,
            "acc_"
            + str(fold)
            + "_"
            + flags.wav2vec2_type
            + "_"
            + str(flags.seed_number)
            + "_"
            + now
            + ".xlsx",
        )
        df.to_excel(filepath, index=False)

        f1scores = np.concatenate(
            (
                f1_osa_train.reshape(flags.n_epochs, 1),
                f1_osa_test.reshape(flags.n_epochs, 1),
            ),
            axis=1,
        )
        df = pd.DataFrame(f1scores, columns=["f1_osa_train", "f1_osa_test"])
        # filepath = os.path.join(acc_dir, 'acc_' + flags.split_mode + '_' + '%d_%f_%d_%d_'
        #                         % (flags.num_modules, flags.alpha_parameter, flags.waiting_epochs,
        #                            flags.maxLength) + 'ad_' + '%d_' % flags.is_adversarial
        #                         + script_name + '_' + now + '.xlsx')
        filepath = os.path.join(
            acc_dir,
            "f1_"
            + str(fold)
            + "_"
            + flags.wav2vec2_type
            + "_"
            + str(flags.seed_number)
            + "_"
            + now
            + ".xlsx",
        )
        df.to_excel(filepath, index=False)

        # if flags.n_epochs >= 30:
        #     print_logging(
        #         'best osa training accuracy in last 30 epochs: {}'.format(max(ac_osa_train[flags.n_epochs - 30:])))
        #     print_logging('best osa test accuracy in last 30 epochs: {}'.format(max(ac_osa_test[flags.n_epochs - 30:])))
        #     print_logging('best osa train f1 score in last 30 epochs: {}'.format(max(f1_osa_train[flags.n_epochs - 30:])))
        #     print_logging('best osa test f1 score in last 30 epochs: {}'.format(max(f1_osa_test[flags.n_epochs - 30:])))

        # if flags.split_mode == 'fold1':
        #     choose_epoch = np.argmax(f1_osa_test[30:]) + 31
        #
        # print_logging('chosen osa training accuracy: {}'.format(ac_osa_train[choose_epoch]))
        # print_logging('chosen osa test accuracy: {}'.format(ac_osa_test[choose_epoch]))
        # print_logging('chosen osa train f1 score: {}'.format(f1_osa_train[choose_epoch]))
        # print_logging('chosen osa test f1 score: {}'.format(f1_osa_test[choose_epoch]))

        print_logging("chosen osa training accuracy: {}".format(ac_osa_train[-1]))
        print_logging("chosen osa test accuracy: {}".format(ac_osa_test[-1]))
        print_logging("chosen osa train f1 score: {}".format(f1_osa_train[-1]))
        print_logging("chosen osa test f1 score: {}".format(f1_osa_test[-1]))
