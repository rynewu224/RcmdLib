
import pandas as pd
import pickle as pkl
import numpy as np
import os
from tqdm import tqdm
from time import time
from collections import defaultdict

item_path = r'C:\work_space\datasets\TaobaoMobile\tianchi_mobile_recommend_train_item.csv'
user_path = r'C:\work_space\datasets\TaobaoMobile\tianchi_mobile_recommend_train_user.csv'
# cols: user_id, item_id, behavior_type, user_geohash, item_category, time

def load_df(path, save_path=''):

    if os.path.exists(save_path):
        print('Loading from '+save_path)
        return pd.read_csv(save_path)

    df = pd.read_csv(path)

    N = df.size()
    df.drop_duplicates(inplace=True)
    print(f'{N-df.size()} duplicate log dropped')

    df = unique_value_to_index(df, 'user_id', '')
    df = unique_value_to_index(df, 'item_id', '')
    df = unique_value_to_index(df, 'user_geohash', '')
    df = unique_value_to_index(df, 'item_category', '')

    if save_path:
        df.to_csv(save_path, index=False)
        print('csv file saved at '+save_path)

    #make_df_profile(df)
    # df.info()
    # print(df.head(20))
    return df

def unique_value_to_index(df: pd.DataFrame, col_name: str, save_dir='../datasets/processed/'):
    """
    把df中的col_name列的值，映射为从1开始的int64 id，自动替换nan为默认值0

    :param df: dataframe
    :param col_name: 要映射的列
    :param save_dir: value2id字典的pkl文件保存路径，若为None，不保存
    :return:
    """
    print(f'Mapping {col_name} to unique id...')

    unique_values = df[col_name].dropna().unique()
    value_to_id = dict()
    id_to_value = dict()

    for i, val in enumerate(unique_values):
        value_to_id[val] = i+1
        id_to_value[i+1] = val

    df[col_name] = df[col_name].apply(lambda x: 0 if pd.isna(x) else value_to_id[x]).astype('int64')

    if save_dir:
        pkl.dump(value_to_id, open(save_dir + f'{col_name}_to_id.pkl', 'wb'), protocol=4)
        pkl.dump(id_to_value, open(save_dir + f'id_to_{col_name}.pkl', 'wb'), protocol=4)
        print('id map files saved at '+save_dir)

    return df

def make_df_profile(df):
    """
    调用pandas_profiling生成df的数据分析报告
    :param df:
    :return:
    """
    import pandas_profiling
    pandas_profiling.ProfileReport(df).to_file('df_report.html')

def filter_behavior(df, behav=0):
    """
    过滤用户行为
    click=1, collect=2, add-to-cart=3, payment=4

    :param df:
    :param behav:
    :return:
    """

    return df[df['behavior_type'] == behav]

def preprocess_uir(df, prepro='origin', level='ui', user_col='user', item_col='item'):
    """
    过滤低频item/user
    例1：prepro='5filter', level='u' 表示过滤交互次数少于5的user
    例2：prepro='3core', level='ui' 表示过滤交互次数少于3的user和item。
        采用计算coreness的方法过滤，遍历u-i图并删除度数=1的节点，执行3次。
    """

    import re
    import gc

    # if prepro.endswith('filter'):
    #     pattern = re.compile(r'\d+')
    #     filter_num = int(pattern.findall(prepro)[0])
    #
    #     tmp1 = df.groupby(['user'], as_index=False)['item'].count()
    #     tmp1.rename(columns={'item': 'cnt_item'}, inplace=True)
    #     tmp2 = df.groupby(['item'], as_index=False)['user'].count()
    #     tmp2.rename(columns={'user': 'cnt_user'}, inplace=True)
    #     df = df.merge(tmp1, on=['user']).merge(tmp2, on=['item'])
    #     if level == 'ui':
    #         df = df.query(f'cnt_item >= {filter_num} and cnt_user >= {filter_num}').reset_index(drop=True).copy()
    #     elif level == 'u':
    #         df = df.query(f'cnt_item >= {filter_num}').reset_index(drop=True).copy()
    #     elif level == 'i':
    #         df = df.query(f'cnt_user >= {filter_num}').reset_index(drop=True).copy()
    #     else:
    #         raise ValueError(f'Invalid level value: {level}')
    #
    #     df.drop(['cnt_item', 'cnt_user'], axis=1, inplace=True)
    #     del tmp1, tmp2
    #     gc.collect()
    #
    # elif prepro.endswith('core'):
    #     pattern = re.compile(r'\d+')
    #     core_num = int(pattern.findall(prepro)[0])
    #
    #     def filter_user(df):
    #         tmp = df.groupby(['user'], as_index=False)['item'].count()
    #         tmp.rename(columns={'item': 'cnt_item'}, inplace=True)
    #         df = df.merge(tmp, on=['user'])
    #         df = df.query(f'cnt_item >= {core_num}').reset_index(drop=True).copy()
    #         df.drop(['cnt_item'], axis=1, inplace=True)
    #
    #         return df
    #
    #     def filter_item(df):
    #         tmp = df.groupby(['item'], as_index=False)['user'].count()
    #         tmp.rename(columns={'user': 'cnt_user'}, inplace=True)
    #         df = df.merge(tmp, on=['item'])
    #         df = df.query(f'cnt_user >= {core_num}').reset_index(drop=True).copy()
    #         df.drop(['cnt_user'], axis=1, inplace=True)
    #
    #         return df
    #
    #     if level == 'ui':
    #         while True:
    #             df = filter_user(df)
    #             df = filter_item(df)
    #             chk_u = df.groupby('user')['item'].count()
    #             chk_i = df.groupby('item')['user'].count()
    #             if len(chk_i[chk_i < core_num]) <= 0 and len(chk_u[chk_u < core_num]) <= 0:
    #                 break
    #     elif level == 'u':
    #         df = filter_user(df)
    #     elif level == 'i':
    #         df = filter_item(df)
    #     else:
    #         raise ValueError(f'Invalid level value: {level}')
    #
    #     gc.collect()


    if prepro.endswith('filter'):
        pattern = re.compile(r'\d+')
        filter_num = int(pattern.findall(prepro)[0])

        # 1. group by user and count item for each user
        tmp1 = df.groupby([user_col], as_index=False)[item_col].count()
        tmp1.rename(columns={item_col: 'cnt_item'}, inplace=True)

        # 2. group by item and count user for each user
        tmp2 = df.groupby([item_col], as_index=False)[user_col].count()
        tmp2.rename(columns={user_col: 'cnt_user'}, inplace=True)

        # 3. join cnt_item and cnt_user to the origin df
        df = df.merge(tmp1, on=[user_col]).merge(tmp2, on=[item_col])

        # 4. filter user/item
        if level == 'ui':
            df = df.query(f'cnt_item >= {filter_num} and cnt_user >= {filter_num}').reset_index(drop=True).copy()
        elif level == 'u':
            df = df.query(f'cnt_item >= {filter_num}').reset_index(drop=True).copy()
        elif level == 'i':
            df = df.query(f'cnt_user >= {filter_num}').reset_index(drop=True).copy()
        else:
            raise ValueError(f'Invalid level value: {level}')

        df.drop(['cnt_item', 'cnt_user'], axis=1, inplace=True)
        del tmp1, tmp2
        gc.collect()

    elif prepro.endswith('core'):
        pattern = re.compile(r'\d+')
        core_num = int(pattern.findall(prepro)[0])

        def filter_user(df):
            tmp = df.groupby([user_col], as_index=False)[item_col].count()
            tmp.rename(columns={item_col: 'cnt_item'}, inplace=True)
            df = df.merge(tmp, on=[user_col])
            df = df.query(f'cnt_item >= {core_num}').reset_index(drop=True).copy()
            df.drop(['cnt_item'], axis=1, inplace=True)

            return df

        def filter_item(df):
            tmp = df.groupby([item_col], as_index=False)[user_col].count()
            tmp.rename(columns={user_col: 'cnt_user'}, inplace=True)
            df = df.merge(tmp, on=[item_col])
            df = df.query(f'cnt_user >= {core_num}').reset_index(drop=True).copy()
            df.drop(['cnt_user'], axis=1, inplace=True)

            return df

        if level == 'ui':
            while True:
                df = filter_user(df)
                df = filter_item(df)
                chk_u = df.groupby(user_col)[item_col].count()
                chk_i = df.groupby(item_col)[user_col].count()
                if len(chk_i[chk_i < core_num]) <= 0 and len(chk_u[chk_u < core_num]) <= 0:
                    break
        elif level == 'u':
            df = filter_user(df)
        elif level == 'i':
            df = filter_item(df)
        else:
            raise ValueError(f'Invalid level value: {level}')

        gc.collect()

    return df


def build_user_all_action_dict(df: pd.DataFrame, save_dir='', action='click'):
    all_save_path = save_dir+f'{action}_all.pkl'
    train_save_path = save_dir+f'{action}_train.pkl'
    test_save_path = save_dir+f'{action}_test.pkl'

    if os.path.exists(train_save_path) and os.path.exists(test_save_path):
        print('Loading from '+save_dir)
        user_train_action = pkl.load(open(train_save_path, 'rb'))
        user_test_action = pkl.load(open(test_save_path, 'rb'))
    else:
        print('Genrating user action dict...')
        user_all_action = defaultdict(list)
        user_train_action = defaultdict(list)
        user_test_action = defaultdict()

        st=time()
        def apply_fn(group):
            user_id = group.values[0][0]
            user_all_action[user_id] = sorted(group.values[:, 1:], key=lambda x: x[-1])
        df.groupby('user_id').apply(apply_fn)    # groupby cost 4.67s for 10k user
        print(f'group by finished in {time()-st} second. ')

        for k in user_all_action:
            user_train_action[k] = user_all_action[k][:-1]
            user_test_action[k] = user_all_action[k][-1]

        pkl.dump(user_all_action, open(all_save_path, 'wb'))
        pkl.dump(user_train_action, open(train_save_path, 'wb'))
        pkl.dump(user_test_action, open(test_save_path, 'wb'))
        print('user action dicts saved at ' + save_dir)

    return user_train_action, user_test_action

"""
features: 
1. user_id
2. user_action (fixed length) (item_id, type, geo, cate, time)
3. user geo
4. item_id
5. item_cate
6. time related(holiday / weekday)

"""





def load_data():
    user_df = load_df(user_path, '../datasets/tianchi_mobile_recommend/processed/tianchi_mobile_rcmd_train_user.csv')
    user_df = filter_behavior(user_df, 1)
    user_df = preprocess_uir(user_df, prepro='5filter', level='u', user_col='user_id', item_col='item_id')
    user_train_action, user_test_action = build_user_all_action_dict(
        user_df,
        save_dir='../datasets/tianchi_mobile_recommend/processed/',
        action='click')
    return user_df, user_train_action, user_test_action


user_df, user_train_action, user_test_action = load_data()