import pandas as pd
import numpy as np
import fasttext

def __main__():
    ## hyperparameters ##
    embeddings_size = 50
    text_transcript = 'data/text_transcript.txt'
    ## main ##
    train = pd.read_csv('data/friends.train.episode_delim.conll',sep='\s+',header=None,comment='#')
    train_scene = pd.read_csv('data/friends.train.scene_delim.conll',sep='\s+',header=None,comment='#')
    trial = pd.read_csv('data/friends.trial.episode_delim.conll',sep='\s+',header=None,comment='#')
    trial_scene = pd.read_csv('data/friends.trial.scene_delim.conll',sep='\s+',header=None,comment='#')
    dfs = [train,train_scene,trial,trial_scene]
    dfs = [clean(df) for df in dfs]
    with open(text_transcript,'wt') as f:
        f.write(make_transcript(dfs))
    embeddings_model = fasttext.skipgram(text_transcript,'embeddings_model',min_count=1,dim=50)
    feature_matrices = []
    for df in dfs:
        feature_matrices.append(make_feature_matrices(train))

def make_feature_matrices(df):
    mentions,mentions_y,mentions_idx = get_mention_arrays(df)
    words,words_idx = get_words(df)
    pre_words = get_pre_words(mentions_idx,words,words_idx)
    next_words = get_next_words(mentions_idx,words,words_idx)
    sents,sents_idx = get_sent_idx(df)
    curr_sents = get_current_sents(mentions_idx,sents,sents_idx)
    next_sents = get_next_sents(mentions_idx,sents,sents_idx)
    pre_sents = get_pre_sents(mentions_idx,sents,sents_idx)
    ut,ut_idx = get_utterances_idx(df)
    curr_ut = get_current_utterance(mentions_idx,ut,ut_idx)
    next_ut = get_next_utterances(mentions_idx,ut,ut_idx)
    pre_ut = get_pre_utterances(mentions_idx,ut,ut_idx)

def make_transcript(dfs,idx=6):
    words = ''
    for df in dfs:
        words += ' '.join(list(df[idx]))
    return words

def clean(df):
    df = df[[0,2,3,4,5,6,9,10,11]]
    return df

def get_mention_rows(df):
    return df[df[11]!='-'][df[11]!='NaN'][df[11]!=np.nan]

def get_words(df):
    l = df[6].values
    return l, range(len(l))

def get_pre_words(mentions_idx,words,words_idx):
    pre_words = []
    for idx in mentions_idx:
        pre_words.append([\
                           words[words_idx[idx[0]-1]] if (idx[-1]-1)>=0 else '',\
                           words[words_idx[idx[0]-2]] if (idx[-1]-2)>=0 else '',\
                           words[words_idx[idx[0]-3]] if (idx[-1]-3)>=0 else ''\
                          ])
    return pre_words

def get_next_words(mentions_idx,words,words_idx):
    next_words = []
    for idx in mentions_idx:
        next_words.append([\
                           words[words_idx[idx[0]+1]] if (idx[-1]+1)<len(words) else '',\
                           words[words_idx[idx[0]+2]] if (idx[-1]+2)<len(words) else '',\
                           words[words_idx[idx[0]+3]] if (idx[-1]+3)<len(words) else ''\
                          ])
    return next_words

def get_sent_idx(df):
    sents = []
    sents_idx = []
    cnt = 0
    for row_no,row in df.iterrows():
        if row[2]==0:
            if row_no!=0:
                cnt +=1
                sents.append(sent_buf)
            sent_buf = row[6]
        else:
            sent_buf += ' ' + row[6]
        sents_idx.append(cnt)
    if df.iloc[len(df)-1][2]!=0:
        sents.append(sent_buf)
        sents_idx.append(cnt)
    return sents,sents_idx

def get_utterances_idx(df):
    ut = []
    ut_idx = []
    cnt = 0
    speaker = None
    for row_no,row in df.iterrows():
        if row[9]!=speaker:
            if row_no!=0:
                ut.append(ut_buf)
                cnt += 1
            speaker = row[9]
            ut_buf = row[6]
        else:
            ut_buf += ' ' + row[6]
        ut_idx.append(cnt)
    if df.iloc[len(df)-1][9]==speaker:
        ut.append(ut_buf)
        ut_idx.append(cnt)
    return ut,ut_idx

def get_mention_arrays(df):
    mentions = []
    mentions_y = []
    mentions_idx = []
    mention_f = False
    for row_no,row in df.iterrows():
        if mention_f and row[11]=='-':
            mention_buf += ' ' + row[6]
        elif row[11]=='-' or type(row[11])==float:
            continue
        elif ('(' in row[11]) and (')' in row[11]):
            mentions.append(row[6])
            mentions_y.append(int(row[11][1:-1]))
            mentions_idx.append([row_no,row_no])
        elif ('(' in row[11]):
            mention_f = True
            mention_buf = row[6]
            mention_idx_buf = row_no
        elif (')' in row[11]):
            mention_f = False
            mention_buf += ' ' + row[6]
            mentions.append(mention_buf)
            mentions_y.append(int(row[11][:-1]))
            mentions_idx.append([mention_idx_buf,row_no])
    return mentions,mentions_y,mentions_idx

def get_current_sents(mentions_idx,sents,sents_idx):
    curr_sents = []
    for idx in mentions_idx:
        curr_sents.append(sents[sents_idx[idx[0]]])
    return curr_sents

def get_next_sents(mentions_idx,sents,sents_idx):
    next_sents = []
    for idx in mentions_idx:
        t = sents_idx[idx[-1]]+1
        if t<len(sents):
            next_sents.append(sents[t])
    return next_sents

def get_pre_sents(mentions_idx,sents,sents_idx):
    pre_sents = []
    for idx in mentions_idx:
        t = sents_idx[idx[0]]
        pre_sents.append([                         sents[t-1] if (t-1)>=0 else '',                          sents[t-2] if (t-2)>=0 else '',                          sents[t-3] if (t-3)>=0 else ''                         ])
    return pre_sents

def get_current_utterance(mentions_idx,ut,ut_idx):
    curr_ut = []
    for idx in mentions_idx:
        curr_ut.append(ut[ut_idx[idx[0]]])
    return curr_ut

def get_next_utterances(mentions_idx,ut,ut_idx):
    next_ut = []
    for idx in mentions_idx:
        t = ut_idx[idx[-1]]+1
        if t<len(ut):
            next_ut.append(ut[t])
    return next_ut

def get_pre_utterances(mentions_idx,ut,ut_idx):
    pre_ut = []
    for idx in mentions_idx:
        t = ut_idx[idx[0]]
        pre_ut.append([                         ut[t-1] if (t-1)>=0 else '',                          ut[t-2] if (t-2)>=0 else '',                          ut[t-3] if (t-3)>=0 else ''                         ])
    return pre_ut

def get_embeddings(l):
    return [embeddings_model[i] for i in l]

def get_avg_embeddings(l):
    a = [np.array([embeddings_model[i] for i in s.split()]) for s in l]
    return [np.sum(i,axis=0)/len(i) for i in a]

def get_multiple_avg_embeddings(l):
    a = [[np.array([embeddings_model[i] for i in sent.split()]) for sent in s] for s in l]
    return [[np.sum(i,axis=0)/len(i) for i in s] for s in a]