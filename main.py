import pandas as pd
import numpy as np
import fasttext

def __main__():
    ## hyperparameters ##
    embeddings_size = 50
    text_transcript = '001.txt'
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
    pairs = []
    for df in dfs:
        pairs.append(make_feature_matrices.append(make_feature_matrices(train)))

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
    phi = [0]*4
    phi[1] = get_phi_1(mentions)
    phi[2] = phi_2(pre_words,next_words,mentions)
    phi[3] = phi_3(pre_sents,next_sents,curr_sents)
    phi[4] = phi_4(pre_ut,next_ut,curr_ut)
    phi[1] = np.reshape(phi_1,[13280,3,50,1])
    phi[2] = np.reshape(phi_2,[13280,7,50,1])
    phi[3] = np.reshape(phi_3,[13280,5,50,1])
    phi[4] = np.reshape(phi_4,[13280,5,50,1])
    pairs = make_mention_pairs(phi,mentions_y)
    return pairs

def get_phi_1(mentions):
    phi = []
    for i in mentions:
        t = i.split()
        if len(t)==0:
            raise ValueError('Empty mention')
        if len(t)==1:
            phi.append(get_embeddings(t+['','']))
        if len(t)==2:
            phi.append(get_embeddings(t+['']))
        else:
            phi.append(get_embeddings(t[0:3]))
    return phi

def get_phi_2(pre_words,next_words,mentions):
    pre = [get_embeddings(i) for i in pre_words]
    suc = [get_embeddings(i) for i in next_words]
    avg_words = get_avg_embeddings(mentions)
    return [np.vstack((i[0],i[1],i[2],j[0],j[1],j[2],k)) for i,j,k in zip(pre,suc,avg)]

def get_phi_3(pre_sents,next_sents,curr_sents):
    p = get_multiple_avg_embeddings(pre_sents)
    s = get_avg_embeddings(next_sents)
    c = get_avg_embeddings(curr_sents)
    return [np.vstack((i[0],i[1],i[2],j,k)) for i,j,k in zip(p,s,c)]

def get_phi_4(pre_ut,next_ut,curr_ut):
    p = get_multiple_avg_embeddings(pre_ut)
    s = get_avg_embeddings(next_ut)
    c = get_avg_embeddings(curr_ut)
    return [np.vstack((i[0],i[1],i[2],j,k)) for i,j,k in zip(p,s,c)]

def make_mention_pairs(phi,mentions_y,window=7):
    tuples = []
    for i in range(len(phi[1])):
        for j in range(i+1,window+i+1):
            if j<len(phi[1]):
                tuples.append([\
                              [phi[k][i] for k in range(1,5)],\
                              [phi[k][j] for k in range(1,5)],\
                              1 if mentions_y[i]==mentions_y[j] else 0\
                             ])
            else:
                break
    return tuples

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
        pre_sents.append([\
                         sents[t-1] if (t-1)>=0 else '',\
                          sents[t-2] if (t-2)>=0 else '',\
                          sents[t-3] if (t-3)>=0 else ''\
                         ])
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
        pre_ut.append([\
                         ut[t-1] if (t-1)>=0 else '',\
                          ut[t-2] if (t-2)>=0 else '',\
                          ut[t-3] if (t-3)>=0 else ''\
                         ])
    return pre_ut

def get_embeddings(l):
    #ip : array of words : n
    #op : array : embedding of each word : n
    return [embeddings_model[i] for i in l]

def get_avg_embeddings(l):
    #ip : array of sentences : n x s
    #op : array : average of embeddings of all words for each sentence : n
    a = [np.array([embeddings_model[i] for i in s.split()]) for s in l]
    return [np.sum(i,axis=0)/len(i) for i in a]

def get_multiple_avg_embeddings(l):
    #ip : array of array of sentences : n x 3 x s
    #op : array of array of average of embeddings of all words for each sentence : n x 3
    ret = []
    for sent_arr in l:
        sent_arr_emb = []
        for sent in sent_arr:
            if sent=='':
                sent_arr_emb.append([0]*embeddings_size)
            else:
                sent_emb = []
                for word in sent.split():
                    sent_emb.append(embeddings_model[word])
                sent_arr_emb.append(np.sum(sent_emb,axis=0)/len(sent_emb))
        ret.append(sent_arr_emb)
    return ret

if __name__=='__main__':
    __main__()