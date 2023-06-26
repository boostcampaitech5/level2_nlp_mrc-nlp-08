import json

import pandas as pd
from tqdm.auto import tqdm


def json_to_csv(path):
    with open(path, "r", encoding='utf-8-sig') as f:
        data = json.load(f)
    df_data = pd.DataFrame({'id':data.keys(), 'answer':data.values()})
    return df_data

def csv_to_json(df_data,path):
    json_dict = dict()
    for i in range(len(df_data)):
        json_dict[df_data['id'].iloc[i]]= df_data['answer'].iloc[i]
    with open(path, "w", encoding='utf-8-sig') as f:
        json.dump(json_dict, f)
    print(f"json file saved in {path}")
    return json_dict

def check_format(s, start, end):
    answer = True
    check=[]
    if start != end : 
        for c in s : 
            if c == start:
                check.append(c)
            elif c == end : 
                if len(check) > 0 : 
                    check.pop()
                else : 
                    answer=False
    elif start == end : 
        for c in s : 
            if c == start : 
                if len(check) > 0 : 
                    check.pop()
                else : 
                    check.append(c)
    if len(check) > 0 : 
        answer = False
    return answer

def change_format(s, start, end):
    s_list = s.split()
    for i in range(len(s_list)):
        if not(check_format(s_list[i], start, end)):
            if len(s_list[i]) == 1 : 
                s_list[i]+=start
            else : 
                if start != end : 
                    if start in s_list[i] : 
                        s_list[i]= s_list[i]+end
                    elif end in s_list[i]:
                        s_list[i]= start + s_list[i]
                else : 
                    if s_list[i][0] == start : 
                        s_list[i]= s_list[i]+end
                    elif s_list[i][1] == end : 
                        s_list[i]= start + s_list[i]
                    else: 
                        s_list[i]= start + s_list[i]
    return " ".join(s_list)

def post_processing(path) : 
    df_data = json_to_csv(path)
    check_dict = {
        "(":")",
        "'":"'",
        "<":">",
        "\"":"\"",
        "《":"》",
        "〈":"〉"
    }

    for i in tqdm(range(len(df_data))):
        for start, end in check_dict.items():
            if not(check_format(df_data['answer'].iloc[i],start,end)) : 
                print(f"{df_data['id'].iloc[i]} : {df_data['answer'].iloc[i]}에서 {start} {end}의 짝이 맞지 않습니다.")
                print("=====Format Changing====")
                print(f"{df_data['answer'].iloc[i]}", end ='')
                df_data['answer'].iloc[i] = change_format(df_data['answer'].iloc[i],start,end)
                print(f" => {df_data['answer'].iloc[i]}")
    csv_to_json(df_data,'output/predictions_eval_postprocessed.json')
def main():
    post_processing('output/predictions.json')

if __name__ =="__main__":
    main()