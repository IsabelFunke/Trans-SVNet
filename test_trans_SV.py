"""
This is an adaptation of trans_SV.py.
The code skips training the model and only performs inference on the validation and test subsets.
Model predictions will be stored in .yaml and (to conform with the TMRNet code) .pkl files.

@Author Isabel Funke, Translational Surgical Oncology, NCT Dresden, Germany
@Credit Xiaojie Gao, Department of Computer Science and Engineering, CUHK Hong Kong, China
@Links https://github.com/xjgaocs/Trans-SVNet/blob/main/trans_SV.py
"""

#some code adapted from https://github.com/YuemingJin/MTRCNet-CL
# and https://github.com/YuemingJin/TMRNet

import torch
from torch import nn
import numpy as np
import pickle, time
import random
from sklearn import metrics
import mstcn
from transformer2_3_1 import Transformer2_3_1
import os
import yaml


def get_data(data_path):
    with open(data_path, 'rb') as f:
        train_test_paths_labels = pickle.load(f)
    # train_paths_19 = train_test_paths_labels[0]
    train_paths_80 = train_test_paths_labels[0]
    val_paths_80 = train_test_paths_labels[1]
    # train_labels_19 = train_test_paths_labels[3]
    train_labels_80 = train_test_paths_labels[2]
    val_labels_80 = train_test_paths_labels[3]
    # train_num_each_19 = train_test_paths_labels[6]
    train_num_each_80 = train_test_paths_labels[4]
    val_num_each_80 = train_test_paths_labels[5]

    test_paths_80 = train_test_paths_labels[6]
    test_labels_80 = train_test_paths_labels[7]
    test_num_each_80 = train_test_paths_labels[8]

    # print('train_paths_19  : {:6d}'.format(len(train_paths_19)))
    # print('train_labels_19 : {:6d}'.format(len(train_labels_19)))
    print('train_paths_80  : {:6d}'.format(len(train_paths_80)))
    print('train_labels_80 : {:6d}'.format(len(train_labels_80)))
    print('valid_paths_80  : {:6d}'.format(len(val_paths_80)))
    print('valid_labels_80 : {:6d}'.format(len(val_labels_80)))

    # train_labels_19 = np.asarray(train_labels_19, dtype=np.int64)
    train_labels_80 = np.asarray(train_labels_80, dtype=np.int64)
    val_labels_80 = np.asarray(val_labels_80, dtype=np.int64)

    train_start_vidx = []
    count = 0
    for i in range(len(train_num_each_80)):
        train_start_vidx.append(count)
        count += train_num_each_80[i]

    val_start_vidx = []
    count = 0
    for i in range(len(val_num_each_80)):
        val_start_vidx.append(count)
        count += val_num_each_80[i]

    test_start_vidx = []
    count = 0
    for i in range(len(test_num_each_80)):
        test_start_vidx.append(count)
        count += test_num_each_80[i]

    return train_labels_80, train_num_each_80, train_start_vidx, val_labels_80, val_num_each_80, val_start_vidx,\
           test_labels_80, test_num_each_80, test_start_vidx

def get_long_feature(start_index, lfb, LFB_length):
    long_feature = []
    long_feature_each = []
    # 上一个存在feature的index
    for k in range(LFB_length):
        LFB_index = (start_index + k)
        LFB_index = int(LFB_index)
        long_feature_each.append(lfb[LFB_index])
    long_feature.append(long_feature_each)
    return long_feature

train_labels_80, train_num_each_80, train_start_vidx,\
    val_labels_80, val_num_each_80, val_start_vidx,\
    test_labels_80, test_num_each_80, test_start_vidx = get_data('./train_val_paths_labels1.pkl')


class Transformer(nn.Module):
    def __init__(self, mstcn_f_maps, mstcn_f_dim, out_features, len_q):
        super(Transformer, self).__init__()
        self.num_f_maps = mstcn_f_maps  # 32
        self.dim = mstcn_f_dim  # 2048
        self.num_classes = out_features  # 7
        self.len_q = len_q

        self.transformer = Transformer2_3_1(d_model=out_features, d_ff=mstcn_f_maps, d_k=mstcn_f_maps,
                                        d_v=mstcn_f_maps, n_layers=1, n_heads=8, len_q = sequence_length)
        self.fc = nn.Linear(mstcn_f_dim, out_features, bias=False)


    def forward(self, x, long_feature):
        out_features = x.transpose(1,2)
        inputs = []
        for i in range(out_features.size(1)):
            if i<self.len_q-1:
                input = torch.zeros((1, self.len_q-1-i, self.num_classes)).cuda()
                input = torch.cat([input, out_features[:, 0:i+1]], dim=1)
            else:
                input = out_features[:, i-self.len_q+1:i+1]
            inputs.append(input)
        inputs = torch.stack(inputs, dim=0).squeeze(1)
        feas = torch.tanh(self.fc(long_feature).transpose(0,1))
        output = self.transformer(inputs, feas)
        #output = output.transpose(1,2)
        #output = self.fc(output)
        return output


with open("./LFB/g_LFB50_train0.pkl", 'rb') as f:
    g_LFB_train = pickle.load(f)

with open("./LFB/g_LFB50_val0.pkl", 'rb') as f:
    g_LFB_val = pickle.load(f)

with open("./LFB/g_LFB50_test0.pkl", 'rb') as f:
    g_LFB_test = pickle.load(f)

print("load completed")

print("g_LFB_train shape:", g_LFB_train.shape)
print("g_LFB_val shape:", g_LFB_val.shape)

tecno_model_name = 'TeCNO50_epoch_6_train_9935_val_8924_test_8603'
transSV_model_name = 'TeCNO50_trans1_3_5_1_length_30_epoch_0_train_8769_val_9054'
exp_name = 'Test_Trans-SVNet'

out_features = 7
num_workers = 3
batch_size = 1
mstcn_causal_conv = True
learning_rate = 1e-3
min_epochs = 12
max_epochs = 25
mstcn_layers = 8
mstcn_f_maps = 32
mstcn_f_dim= 2048
mstcn_stages = 2

sequence_length = 30

seed = 1
print("Random Seed: ", seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

num_gpu = torch.cuda.device_count()
use_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if use_gpu else "cpu")

model = mstcn.MultiStageModel(mstcn_stages, mstcn_layers, mstcn_f_maps, mstcn_f_dim, out_features, mstcn_causal_conv)
model_path = './best_model/TeCNO/'
model.load_state_dict(torch.load(model_path+tecno_model_name+'.pth'))
model.cuda()
model.eval()

model1 = Transformer(mstcn_f_maps, mstcn_f_dim, out_features, sequence_length)
model1.load_state_dict(torch.load(model_path+transSV_model_name+'.pth'))
model1.cuda()
model1.eval()

# train_we_use_start_idx_80 = [x for x in range(40)]
val_we_use_start_idx_80 = [x for x in range(8)]
test_we_use_start_idx_80 = [x for x in range(32)]
"""
train video ids: 1 ... 40
val video ids: 41 ... 48
test video ids: 49 ... 80
"""
start_val = 41
start_test = 49

pred_40_8_32 = {  # predictions on 32 test videos
    'fps': 1,
    'TeCNO_model': tecno_model_name,
    'TransSV_model': transSV_model_name
}
pred_40_40 = {  # predictions on 8 validation + 32 test videos
    'fps': 1,
    'TeCNO_model': tecno_model_name,
    'TransSV_model': transSV_model_name
}

# results on validation subset

val_corrects_phase = 0
val_start_time = time.time()
val_progress = 0
val_all_preds_phase = []
val_all_labels_phase = []
val_acc_each_video = []

with torch.no_grad():
    for i in val_we_use_start_idx_80:
        labels_phase = []
        for j in range(val_start_vidx[i], val_start_vidx[i] + val_num_each_80[i]):
            labels_phase.append(val_labels_80[j][0])
        labels_phase = torch.LongTensor(labels_phase)
        if use_gpu:
            labels_phase = labels_phase.to(device)
        else:
            labels_phase = labels_phase

        long_feature = get_long_feature(start_index=val_start_vidx[i],
                                        lfb=g_LFB_val, LFB_length=val_num_each_80[i])

        long_feature = (torch.Tensor(long_feature)).to(device)
        video_fe = long_feature.transpose(2, 1)

        out_features = model.forward(video_fe)[-1]
        out_features = out_features.squeeze(1)
        p_classes1 = model1(out_features, long_feature)

        p_classes = p_classes1.squeeze()

        _, preds_phase = torch.max(p_classes.data, 1)
        pred_40_40["{:02d}".format(i + start_val)] = preds_phase.int().tolist()

        val_corrects_phase += torch.sum(preds_phase == labels_phase.data)
        val_acc_each_video.append(float(torch.sum(preds_phase == labels_phase.data))/val_num_each_80[i])
        # TODO

        for j in range(len(preds_phase)):
            val_all_preds_phase.append(int(preds_phase.data.cpu()[j]))
        for j in range(len(labels_phase)):
            val_all_labels_phase.append(int(labels_phase.data.cpu()[j]))

        val_progress += 1
        if val_progress * batch_size >= len(val_we_use_start_idx_80):
            percent = 100.0
            print('Val progress: %s [%d/%d]' % (str(percent) + '%', len(val_we_use_start_idx_80),
                                                len(val_we_use_start_idx_80)), end='\n')
        else:
            percent = round(val_progress * batch_size / len(val_we_use_start_idx_80) * 100, 2)
            print('Val progress: %s [%d/%d]' % (str(percent) + '%', val_progress * batch_size, len(val_we_use_start_idx_80)),
                  end='\r')

#evaluation only for training reference
val_elapsed_time = time.time() - val_start_time
val_accuracy_phase = float(val_corrects_phase) / len(val_labels_80)
val_acc_video = np.mean(val_acc_each_video)

val_recall_phase = metrics.recall_score(val_all_labels_phase, val_all_preds_phase, average='macro')
val_precision_phase = metrics.precision_score(val_all_labels_phase, val_all_preds_phase, average='macro')
val_jaccard_phase = metrics.jaccard_score(val_all_labels_phase, val_all_preds_phase, average='macro')
val_precision_each_phase = metrics.precision_score(val_all_labels_phase, val_all_preds_phase, average=None)
val_recall_each_phase = metrics.recall_score(val_all_labels_phase, val_all_preds_phase, average=None)

# results on test subset

test_progress = 0
test_corrects_phase = 0
test_all_preds_phase = []
test_all_labels_phase = []
test_acc_each_video = []
test_start_time = time.time()

with torch.no_grad():
    for i in test_we_use_start_idx_80:
        labels_phase = []
        for j in range(test_start_vidx[i], test_start_vidx[i] + test_num_each_80[i]):
            labels_phase.append(test_labels_80[j][0])
        labels_phase = torch.LongTensor(labels_phase)
        if use_gpu:
            labels_phase = labels_phase.to(device)
        else:
            labels_phase = labels_phase

        long_feature = get_long_feature(start_index=test_start_vidx[i],
                                        lfb=g_LFB_test, LFB_length=test_num_each_80[i])

        long_feature = (torch.Tensor(long_feature)).to(device)
        video_fe = long_feature.transpose(2, 1)

        out_features = model.forward(video_fe)[-1]
        out_features = out_features.squeeze(1)
        p_classes1 = model1(out_features, long_feature)

        p_classes = p_classes1.squeeze()

        _, preds_phase = torch.max(p_classes.data, 1)
        pred_40_40["{:02d}".format(i + start_test)] = preds_phase.int().tolist()
        pred_40_8_32["{:02d}".format(i + start_test)] = preds_phase.int().tolist()

        test_corrects_phase += torch.sum(preds_phase == labels_phase.data)
        test_acc_each_video.append(float(torch.sum(preds_phase == labels_phase.data)) / test_num_each_80[i])
        # TODO

        for j in range(len(preds_phase)):
            test_all_preds_phase.append(int(preds_phase.data.cpu()[j]))
        for j in range(len(labels_phase)):
            test_all_labels_phase.append(int(labels_phase.data.cpu()[j]))

        test_progress += 1
        if test_progress * batch_size >= len(test_we_use_start_idx_80):
            percent = 100.0
            print('Test progress: %s [%d/%d]' % (str(percent) + '%', len(test_we_use_start_idx_80),
                                                len(test_we_use_start_idx_80)), end='\n')
        else:
            percent = round(test_progress * batch_size / len(test_we_use_start_idx_80) * 100, 2)
            print('Test progress: %s [%d/%d]' % (
            str(percent) + '%', test_progress * batch_size, len(test_we_use_start_idx_80)),
                  end='\r')

test_accuracy_phase = float(test_corrects_phase) / len(test_labels_80)
test_acc_video = np.mean(test_acc_each_video)
test_elapsed_time = time.time() - test_start_time

test_recall_phase = metrics.recall_score(test_all_labels_phase, test_all_preds_phase, average='macro')
test_precision_phase = metrics.precision_score(test_all_labels_phase, test_all_preds_phase, average='macro')
test_jaccard_phase = metrics.jaccard_score(test_all_labels_phase, test_all_preds_phase, average='macro')
test_precision_each_phase = metrics.precision_score(test_all_labels_phase, test_all_preds_phase, average=None)
test_recall_each_phase = metrics.recall_score(test_all_labels_phase, test_all_preds_phase, average=None)

print('valid in: {:2.0f}m{:2.0f}s'
      ' valid accu(phase): {:.4f}'
      ' valid accu(video): {:.4f}'
      ' test in: {:2.0f}m{:2.0f}s'
      ' test accu(phase): {:.4f}'
      ' test accu(video): {:.4f}'
      .format(val_elapsed_time // 60,
              val_elapsed_time % 60,
              val_accuracy_phase,
              val_acc_video,
              test_elapsed_time // 60,
              test_elapsed_time % 60,
              test_accuracy_phase,
              test_acc_video))

print("val_precision_each_phase:", val_precision_each_phase)
print("val_recall_each_phase:", val_recall_each_phase)
print("val_precision_phase", val_precision_phase)
print("val_recall_phase", val_recall_phase)
print("val_jaccard_phase", val_jaccard_phase)

print("test_precision_each_phase:", test_precision_each_phase)
print("test_recall_each_phase:", test_recall_each_phase)
print("test_precision_phase", test_precision_phase)
print("test_recall_phase", test_recall_phase)
print("test_jaccard_phase", test_jaccard_phase)

# store predictions

for pred_dict, out_dir in [(pred_40_40, os.path.join('./Eval', exp_name, "40-40", "-")),
                           (pred_40_8_32, os.path.join('./Eval', exp_name, "40-8-32", "-"))]:
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    yaml.safe_dump(pred_dict, stream=open(os.path.join(out_dir, "predictions.yaml"), 'w'), default_flow_style=True)

all_predictions = val_all_preds_phase + test_all_preds_phase
pkl_path = os.path.join('./Eval', exp_name, "all_predictions.pkl")
with open(pkl_path, 'wb') as f:
    pickle.dump(all_predictions, f)
