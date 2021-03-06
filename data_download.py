import pandas as pd
import numpy as np
import pandas as pd

print("Data loading")
df =pd.read_table('data.tsv',header=None)
df.columns =['query_id','ques','pass','label_passage','passage_sequence']
print("Data loading finished")

def under_sampling(train_df):
    num_correct_label = len(train_df[train_df['label_passage']==1])
    incorrect_indices = train_df[train_df.label_passage==0].index
    random_indices = np.random.choice(incorrect_indices,num_correct_label,replace = False)
    print(random_indices.sum())
    correct_indices = train_df[train_df.label_passage ==1].index
    under_sample_indices = np.concatenate([correct_indices, random_indices])
    under_sample = train_df.loc[under_sample_indices]
    under_sample =under_sample.sample(frac=1,random_state=20).reset_index(drop= True)
    return under_sample

under_sample_1 = under_sampling(df)
under_sample_1 = under_sample_1.sample(30000)
under_sample_1.to_csv('glue_data/MRPC/train.csv')

under_sample_2 = under_sampling(df)
under_sample_2 = under_sample_2.sample(10000)
under_sample_2.to_csv('glue_data/MRPC/dev.csv')

eval_data = pd.read_table("eval1_unlabelled.tsv", header=None)
eval_data.to_csv("glue_data/MRPC/test.csv")

