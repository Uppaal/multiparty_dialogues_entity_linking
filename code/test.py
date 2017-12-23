# from code.model_cleaned import ACNN
import numpy as np
import pickle as pkl
from sklearn.metrics import average_precision_score
from model_cleaned import ACNN

train = False
if train:
    t_data = pkl.load(open("../data/training_split",'rb'))
    np_phim1_1_train = t_data[0]
    np_phim1_2_train = t_data[1]
    np_phim1_3_train = t_data[2]
    np_phim1_4_train = t_data[3]
    np_phim1_d_train = t_data[4]
    np_phim2_1_train = t_data[5]
    np_phim2_2_train = t_data[6]
    np_phim2_3_train = t_data[7]
    np_phim2_4_train = t_data[8]
    np_phim2_d_train = t_data[9]
    np_phi_p_train   = t_data[10]
    np_labels_train  = t_data[11]

    train_len = len(np_phim1_1_train)
    batch_size = 2000
    num_batches = int(float(train_len / batch_size) + 0.5)
    assert num_batches != 0
    num_epochs = 3000
    model = ACNN(2)
    best_cost = 99999999999999999.0
    for epoch in range(num_epochs):
        epoch_cost = 0.0
        for i in range(num_batches):
            _, c = model.sess.run([model.train_op, model.loss],
                                  feed_dict={model.phim1_1: np_phim1_1_train[i * batch_size: (i + 1) * batch_size],
                                             model.phim1_2: np_phim1_2_train[i * batch_size: (i + 1) * batch_size],
                                             model.phim1_3: np_phim1_3_train[i * batch_size: (i + 1) * batch_size],
                                             model.phim1_4: np_phim1_4_train[i * batch_size: (i + 1) * batch_size],
                                             model.phim1_d: np_phim1_d_train[i * batch_size: (i + 1) * batch_size],
                                             model.phim2_1: np_phim2_1_train[i * batch_size: (i + 1) * batch_size],
                                             model.phim2_2: np_phim2_2_train[i * batch_size: (i + 1) * batch_size],
                                             model.phim2_3: np_phim2_3_train[i * batch_size: (i + 1) * batch_size],
                                             model.phim2_4: np_phim2_4_train[i * batch_size: (i + 1) * batch_size],
                                             model.phim2_d: np_phim2_d_train[i * batch_size: (i + 1) * batch_size],
                                             model.phi_p:   np_phi_p_train[i * batch_size: (i + 1) * batch_size],
                                             model.label:   np_labels_train[i * batch_size: (i + 1) * batch_size]})
            print("BATCH_COST:%f" %c)
            epoch_cost += c
        epoch_cost = epoch_cost/batch_size
        if epoch%10 == 0:
            print ("EPOCH:%d    COST:%f"%(epoch, epoch_cost))
            if epoch_cost < best_cost:
                best_cost = epoch_cost
                print("SAVING MODEL")
                save_path = model.saver.save(model.sess, "../saved_models/trainedMODEL.ckpt")
                print("MODEL SAVED IN %s"%save_path)
else:
    test_data = pkl.load(open("../data/test_split",'rb'))
    print (len(test_data))
    np_phim1_1_test = test_data[0]
    np_phim1_2_test = test_data[1]
    np_phim1_3_test = test_data[2]
    np_phim1_4_test = test_data[3]
    np_phim1_d_test = test_data[4]
    np_phim2_1_test = test_data[5]
    np_phim2_2_test = test_data[6]
    np_phim2_3_test = test_data[7]
    np_phim2_4_test = test_data[8]
    np_phim2_d_test = test_data[9]
    np_phi_p_test   = test_data[10]
    np_labels_test  = test_data[11]


    tes_data = [np_phim1_1_test, np_phim1_2_test, np_phim1_3_test, np_phim1_4_test, np_phim1_d_test, np_phim2_1_test,
             np_phim2_2_test, np_phim2_3_test, np_phim2_4_test, np_phim2_d_test, np_phi_p_test]


    model1 = ACNN(2, mode='test')
    model1.saver.restore(model1.sess, "../saved_models/NLP_Blake/trainedMODEL_newrun.ckpt")

    Xtest = [np_phim1_1_test, np_phim1_2_test, np_phim1_3_test, np_phim1_4_test, np_phim1_d_test, np_phim2_1_test,
             np_phim2_2_test,
             np_phim2_3_test, np_phim2_4_test, np_phim2_d_test, np_phi_p_test]
    Ytest = np_labels_test

    y = model1.predict(Xtest)

    print((sum(y == Ytest))/float(len(Ytest)))
    print (average_precision_score(Ytest,y))
    print("OK")
