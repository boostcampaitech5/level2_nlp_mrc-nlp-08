import pandas as pd
from datasets import load_dataset
import numpy as np


# 데이터셋 로드
dataset = load_dataset('squad_kor_v1')

# 데이터프레임으로 변환
df = pd.DataFrame(dataset['train'])
df2 = pd.DataFrame(dataset['validation'])
df = pd.concat([df,df2])
#df = pd.read_csv('./csv_data/train_data.csv')
np_df = np.array(df)
title_list = []
context_list = []
answer_list = []
question_list = []
id_list = []
document_id_list = []
index_level_0_list = []

count = 0
for i in range(len(np_df)):
    #print(np_df[i][4])
    temp_dict = {}
    temp_dict['answer_start'] = np_df[i][4]['answer_start']
    temp_dict['text'] = np_df[i][4]['text']
    title_list.append(np_df[i][1])
    context_list.append(np_df[i][2])
    question_list.append(np_df[i][3])
    id_list.append(np_df[i][0])
    answer_list.append(temp_dict)


df = pd.DataFrame({
    'title' : title_list,
    'context' : context_list,
    'question' : question_list,
    'id' : id_list,
    'answers' : answer_list,
    'document_id' : [i for i in range(len(id_list))],
    '__index_level_0__' : [i for i in range(len(id_list))]
})
# CSV 파일로 저장
df.to_csv('./csv_data/squad_kor_v1.csv', index=False)
