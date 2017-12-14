# from code.model_cleaned import ACNN
import numpy as np
import pickle as pkl

data_all = pkl.load(open("D:/temp_data/train_tuples.pickle", "rb"))
phi_p = pkl.load(open("../data/phi_p.pkl", 'rb'))
temp_phi_p = []
for every in phi_p:
    val = []
    for each in every:
        val.append(each)
    temp_phi_p.append(np.asarray(val))

np_phi_p = np.asarray(temp_phi_p)
# Datasets
phim1_1 = []
phim1_2 = []
phim1_3 = []
phim1_4 = []
phim1_d = []
phim1_p = []

phim2_1 = []
phim2_2 = []
phim2_3 = []
phim2_4 = []
phim2_d = []
phim2_p = []
# Labels
labels = []

for i, each in enumerate(data_all):
    phim1_1.append(each[0][0])
    phim1_2.append(each[0][1])
    phim1_3.append(each[0][2])
    phim1_4.append(each[0][3])
    phim1_d.append(each[0][4])

    phim2_1.append(each[1][0])
    phim2_2.append(each[1][1])
    phim2_3.append(each[1][2])
    phim2_4.append(each[1][3])
    phim2_d.append(each[1][4])

    labels.append(each[2])

np_phim1_1 = np.asarray(phim1_1)
np_phim1_2 = np.asarray(phim1_2)
np_phim1_3 = np.asarray(phim1_3)
np_phim1_4 = np.asarray(phim1_4)
np_phim1_d = np.asarray(phim1_d)
np_phim2_1 = np.asarray(phim2_1)
np_phim2_2 = np.asarray(phim2_2)
np_phim2_3 = np.asarray(phim2_3)
np_phim2_4 = np.asarray(phim2_4)
np_phim2_d = np.asarray(phim2_d)
np_labels = np.asarray(labels)

total_data_len = len(np_phim1_1)
train_len = int(0.8 * total_data_len)
test_len = total_data_len - train_len

np_phim1_1_train = np_phim1_1[0:train_len]
np_phim1_2_train = np_phim1_2[0:train_len]
np_phim1_3_train = np_phim1_3[0:train_len]
np_phim1_4_train = np_phim1_4[0:train_len]
np_phim1_d_train = np_phim1_d[0:train_len]

np_phim2_1_train = np_phim2_1[0:train_len]
np_phim2_2_train = np_phim2_2[0:train_len]
np_phim2_3_train = np_phim2_3[0:train_len]
np_phim2_4_train = np_phim2_4[0:train_len]
np_phim2_d_train = np_phim2_d[0:train_len]

np_phi_p_train = np_phi_p[0:train_len]
np_labels_train = np_labels[0:train_len]
np_labels_train = np_labels_train[..., np.newaxis]

t_data = [np_phim1_1_train,
np_phim1_2_train,
np_phim1_3_train,
np_phim1_4_train,
np_phim1_d_train,
np_phim2_1_train,
np_phim2_2_train,
np_phim2_3_train,
np_phim2_4_train,
np_phim2_d_train,
np_phi_p_train  ,
np_labels_train]



batch_size = 200
num_batches = int(float(train_len / batch_size) + 0.5)
assert num_batches != 0
num_epochs = 3000
# model = ACNN(2)
# best_cost = 99999999999999999.0
# for epoch in range(num_epochs):
#     epoch_cost = 0.0
#     for i in range(num_batches):
#         _, c = model.sess.run([model.train_op, model.loss],
#                               feed_dict={model.phim1_1: np_phim1_1_train[i * batch_size: (i + 1) * batch_size],
#                                          model.phim1_2: np_phim1_2_train[i * batch_size: (i + 1) * batch_size],
#                                          model.phim1_3: np_phim1_3_train[i * batch_size: (i + 1) * batch_size],
#                                          model.phim1_4: np_phim1_4_train[i * batch_size: (i + 1) * batch_size],
#                                          model.phim1_d: np_phim1_d_train[i * batch_size: (i + 1) * batch_size],
#                                          model.phim2_1: np_phim2_1_train[i * batch_size: (i + 1) * batch_size],
#                                          model.phim2_2: np_phim2_2_train[i * batch_size: (i + 1) * batch_size],
#                                          model.phim2_3: np_phim2_3_train[i * batch_size: (i + 1) * batch_size],
#                                          model.phim2_4: np_phim2_4_train[i * batch_size: (i + 1) * batch_size],
#                                          model.phim2_d: np_phim2_d_train[i * batch_size: (i + 1) * batch_size],
#                                          model.phi_p:   np_phi_p_train[i * batch_size: (i + 1) * batch_size],
#                                          model.label:   np_labels_train[i * batch_size: (i + 1) * batch_size]})
#         epoch_cost += c
#     if epoch%100 == 0:
#         print ("EPOCH:%d    COST:%f"%(epoch, epoch_cost))
#         if epoch_cost < best_cost:
#             best_cost = epoch_cost
#             print("SAVING MODEL")
#             save_path = model.saver.save(model.sess, "../saved_models/trainedMODEL.ckpt")
#             print("MODEL SAVED IN %s"%save_path)

np_phim1_1_test = np_phim1_1[train_len:]
np_phim1_2_test = np_phim1_2[train_len:]
np_phim1_3_test = np_phim1_3[train_len:]
np_phim1_4_test = np_phim1_4[train_len:]
np_phim2_1_test = np_phim2_1[train_len:]
np_phim2_2_test = np_phim2_2[train_len:]
np_phim2_3_test = np_phim2_3[train_len:]
np_phim2_4_test = np_phim2_4[train_len:]

np_phim1_d_test = np_phim1_d[train_len:]
np_phim2_d_test = np_phim2_d[train_len:]
np_phi_p_test = np_phi_p[train_len:]
np_labels_test = np_labels[train_len:]


tes_data = [np_phim1_1_test, np_phim1_2_test, np_phim1_3_test, np_phim1_4_test, np_phim1_d_test, np_phim2_1_test,
         np_phim2_2_test, np_phim2_3_test, np_phim2_4_test, np_phim2_d_test, np_phi_p_test]


# model1 = ACNN(2,)
# model1.saver.restore(model1.sess, "../saved_models/trainedMODEL.ckpt")
#
# Xtest = [np_phim1_1_test, np_phim1_2_test, np_phim1_3_test, np_phim1_4_test, np_phim1_d_test, np_phim2_1_test,
#          np_phim2_2_test,
#          np_phim2_3_test, np_phim2_4_test, np_phim2_d_test, np_phi_p_test]
# Ytest = np_labels_test
#
# y = model1.predict(Xtest)
# print(sum(y == Ytest))
# print("OK")
