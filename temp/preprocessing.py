import pandas as pd
import numpy as np
import fasttext
import stringdist
import re

global data_path

data_path = "../data/"

def __main__():
    embeddings_size = 50
    text_transcript = data_path + 'text_transcript.txt'
    
    train = pd.read_csv(data_path + 'friends.train.episode_delim.conll',sep='\s+',header=None,comment='#')
    train_scene = pd.read_csv(data_path + 'friends.train.scene_delim.conll',sep='\s+',header=None,comment='#')
    trial = pd.read_csv(data_path + 'friends.trial.episode_delim.conll',sep='\s+',header=None,comment='#')
    trial_scene = pd.read_csv(data_path + 'friends.trial.scene_delim.conll',sep='\s+',header=None,comment='#')
    # dfs = [train,train_scene,trial,trial_scene]
    dfs = [train]
    
    with open(text_transcript,'wt') as f:
       f.write(make_transcript(dfs))
    embeddings_model = fasttext.skipgram(text_transcript,'embeddings_model',min_count=1,dim=embeddings_size)
    # embeddings_model = fasttext.load_model('embeddings_model.bin')
    print("Embeddings trained")
    
    all_pairs = []
    all_phi_p = []
    for df_idx,df in enumerate(dfs):
        mentions,mentions_y,mentions_idx = get_mention_arrays(df)
        print('Mentions done')
        words,words_idx = get_words_idx(df)
        pre_words = get_pre_words(mentions_idx,words,words_idx)
        next_words = get_next_words(mentions_idx,words,words_idx)
        print('words done')
        sents,sents_idx = get_sent_idx(df)
        curr_sents = get_current_sents(mentions_idx,sents,sents_idx)
        next_sents = get_next_sents(mentions_idx,sents,sents_idx)
        pre_sents = get_pre_sents(mentions_idx,sents,sents_idx)
        print('sents done')
        ut,ut_idx = get_utterances_idx(df)
        curr_ut = get_current_utterance(mentions_idx,ut,ut_idx)
        next_ut = get_next_utterances(mentions_idx,ut,ut_idx)
        pre_ut = get_pre_utterances(mentions_idx,ut,ut_idx)
        print('ut done')
        speakers,speakers_idx = get_speakers_idx(df)
        curr_speakers = get_current_speakers(mentions_idx,speakers,speakers_idx)
        pre_speakers = get_pre_speakers(mentions_idx,speakers,speakers_idx)
        print('speakers done')
        gender_info = get_gender_info(mentions)
        print('gender info done')
        phi = [0]*6
        phi[1] = get_phi_1(mentions,embeddings_model)
        phi[2] = get_phi_2(pre_words,next_words,mentions,embeddings_model)
        phi[3] = get_phi_3(pre_sents,next_sents,curr_sents,embeddings_model)
        phi[4] = get_phi_4(pre_ut,next_ut,curr_ut,embeddings_model)
        phi[5] = get_phi_d(curr_speakers,pre_speakers,gender_info,embeddings_model)
        phi[1] = np.reshape(phi[1],[-1,3,embeddings_size,1])
        phi[2] = np.reshape(phi[2],[-1,7,embeddings_size,1])
        phi[3] = np.reshape(phi[3],[-1,5,embeddings_size,1])
        phi[4] = np.reshape(phi[4],[-1,5,embeddings_size,1])
        phi[5] = np.reshape(phi[5],[-1,embeddings_size*3+4,1])
        print('phi done')
        pairs = make_mention_pairs(phi,mentions_y)
        print('pairs done')
        np.save('pairs_'+df_idx+'.npy',pairs)
        all_pairs.append(pairs)
        phi_p = get_phi_p(mentions,curr_speakers,curr_sents)
        np.save('phi_p_'+df_idx+'.npy',phi_p)
        all_phi_p.append(phi_p)
    # all_pairs = np.array(all_pairs)
    # np.save('pairs.npy',all_pairs)
    
def make_mention_pairs(phi,mentions_y,window=7):
    tuples = []
    for i in range(len(phi[1])):
        for j in range(i+1,window+i+1):
            if j<len(phi[1]):
                tuples.append([\
                              [phi[k][i] for k in range(1,6)],\
                              [phi[k][j] for k in range(1,6)],\
                              1 if mentions_y[i]==mentions_y[j] else 0\
                             ])
            else:
                break
    return tuples
    
def get_phi_1(mentions,embeddings_model):
    phi = []
    for i in mentions:
        t = i.split()
        if len(t)==0:
            raise ValueError('Empty mention')
        elif len(t)==1:
            phi.append(get_embeddings(t+['',''],embeddings_model))
        elif len(t)==2:
            phi.append(get_embeddings(t+[''],embeddings_model))
        else:
            phi.append(get_embeddings(t[0:3],embeddings_model))
    return np.array(phi)

def get_phi_2(pre_words,next_words,mentions,embeddings_model):
    pre = [get_embeddings(i,embeddings_model) for i in pre_words]
    suc = [get_embeddings(i,embeddings_model) for i in next_words]
    avg = get_avg_embeddings(mentions,embeddings_model)
    return np.array([np.vstack((i[0],i[1],i[2],j[0],j[1],j[2],k)) for i,j,k in zip(pre,suc,avg)])

def get_phi_3(pre_sents,next_sents,curr_sents,embeddings_model):
    p = get_multiple_avg_embeddings(pre_sents,embeddings_model)
    s = get_avg_embeddings(next_sents,embeddings_model)
    c = get_avg_embeddings(curr_sents,embeddings_model)
    return np.array([np.vstack((i[0],i[1],i[2],j,k)) for i,j,k in zip(p,s,c)])

def get_phi_4(pre_ut,next_ut,curr_ut,embeddings_model):
    p = get_multiple_avg_embeddings(pre_ut,embeddings_model)
    s = get_avg_embeddings(next_ut,embeddings_model)
    c = get_avg_embeddings(curr_ut,embeddings_model)
    return np.array([np.vstack((i[0],i[1],i[2],j,k)) for i,j,k in zip(p,s,c)])

def get_phi_d(curr_speakers,pre_speakers,gender_info,embeddings_model):
    curr_s = get_embeddings(curr_speakers,embeddings_model)
    pre_s = [get_embeddings(i,embeddings_model) for i in pre_speakers]
    return np.array([list(i)+list(j[0])+list(j[1])+list(k) for i,j,k in zip(curr_s,pre_s,gender_info)])

def get_phi_p(mentions,speakers,sentences,window=7):
    """ returns normalized mention_pair features """
    def exact_string_match(m1,m2):
        return int(m1==m2)
    def speaker_match(s1,s2):
        return int(s1==s2)
    def edit_distance(m1,m2):
        return stringdist.levenshtein(m1,m2)
    tuples = []
    for i in range(len(mentions)):
        for j in range(i+1,window+i+1):
            if j<len(mentions):
                tuples.append([exact_string_match(mentions[i],mentions[j]),\
                               speaker_match(speakers[i],speakers[j]),\
                               edit_distance(mentions[i],mentions[j]),\
                               edit_distance(sentences[i],sentences[j])])
            else:
                break
    tuples = np.array(tuples)
    return np.divide(tuples,np.sum(tuples,axis=0))

def get_mention_rows(df):
    return df[df[11]!='-'][df[11]!='NaN'][df[11]!=np.nan]

def get_words_idx(df):
    l = df[6].values
    return l, df.index.values

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

def get_speakers_idx(df):
    speaker = None
    speakers = []
    speakers_idx = []
    cnt = -1
    for row_no,row in df.iterrows():
        c_speaker = row[9] if row[9]==row[9] else ''
        if c_speaker!=speaker:
            speakers.append(c_speaker)
            cnt += 1
            speaker = c_speaker
        speakers_idx.append(cnt)
    return speakers, speakers_idx

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

def get_current_speakers(mentions_idx,speakers,speakers_idx):
    curr_speakers = []
    for idx in mentions_idx:
        curr_speakers.append(speakers[speakers_idx[idx[0]]])
    return curr_speakers

def get_pre_speakers(mentions_idx,speakers,speakers_idx):
    print("new")
    pre_speakers = []
    for idx in mentions_idx:
        pre_speakers.append([\
                           speakers[speakers_idx[idx[0]]-1] if (speakers_idx[idx[0]]-1)>=0 else '',\
                           speakers[speakers_idx[idx[0]]-2] if (speakers_idx[idx[0]]-2)>=0 else ''\
                          ])
    return pre_speakers

def get_gender_info(mentions):
    with open(data_path+'gender.data','rb') as f:
        gender_data = [i.decode().strip().split('\t') for i in f.readlines()]
    gender_words = {re.sub('[^\w\s]','',i[0].lower()).strip():np.array([int(j) for j in i[1].split(' ')]) for i in gender_data}
    return np.array([np.mean(np.array([gender_words[word] if word in gender_words else np.zeros(4) for word in mention.split()]),axis=0) for mention in mentions])

def get_embeddings(l,embeddings_model):
    """ip : array of words : n
    op : array : embedding of each word : n"""
    return np.array([embeddings_model[i] for i in l])

def get_avg_embeddings(l,embeddings_model):
    """ip : array of sentences : n x s
    op : array : average of embeddings of all words for each sentence : n"""
    a = [np.array([embeddings_model[i] for i in sent.split()]) for sent in l]
    return np.array([np.sum(i,axis=0)/len(i) for i in a])

def get_multiple_avg_embeddings(l,embeddings_model):
    """ip : array of array of sentences : n x 3 x s
    op : array of array of average of embeddings of all words for each sentence : n x 3"""
    ret = []
    for sent_arr in l:
        sent_arr_emb = []
        for sent in sent_arr:
            if sent=='':
                sent_arr_emb.append(embeddings_model[''])
            else:
                sent_emb = []
                for word in sent.split():
                    sent_emb.append(embeddings_model[word])
                sent_arr_emb.append(np.sum(sent_emb,axis=0)/len(sent_emb))
        ret.append(sent_arr_emb)
    return np.array(ret)

def make_transcript(dfs,idx=6):
    words = ''
    for df in dfs:
        words += ' '.join(list(df[idx]))
    return words

def clean(df):
    df = df[[0,2,3,4,5,6,9,10,11]]
    return df

if __name__=='__main__':
    __main__()