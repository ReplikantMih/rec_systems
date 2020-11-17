def prefilter_items(data, item_features):
    
    data = data.merge(item_features, how='left', on='item_id')
    
    # ������ ����� ���������� ������ (�� � ��� �����)
    popularity = data.groupby('item_id')['user_id'].nunique().reset_index()
    popularity['user_id'] = popularity['user_id']/ data['user_id'].nunique()
    
    popularity.rename(columns={'user_id': 'share_unique_users'}, inplace=True)
    popularity.sort_values('share_unique_users', inplace=True)
    
    top_popular = popularity[popularity['share_unique_users'] > 0.5].item_id.tolist()
    data = data[~data['item_id'].isin(top_popular)]
    
    # ������ ����� �� ���������� ������ (�� � ��� �� �����)
    top_notpopular = popularity[popularity['share_unique_users'] < 0.01].item_id.tolist()
    data = data[~data['item_id'].isin(top_notpopular)]
    
    # ������ ������, ������� �� ����������� �� ��������� 12 �������
    data = data[~(data['week_no'] > data['week_no'].max() - 12 * 4)]
    
    # ������ �� ���������� ��� ������������� ��������� (department)
    deps_to_delete = ['GROCERY', 'MISC. TRANS.', 'PASTRY', 'DRUG GM', 'MEAT-PCKGD',
       'SEAFOOD-PCKGD', 'PRODUCE', 'NUTRITION', 'DELI', 'COSMETICS']
    data = data[~(data['department'].isin(deps_to_delete))]
    
    # ������ ������� ������� ������ (�� ��� �� ����������). 1 ������� �� �������� ����� 60 ���. 
    data = data[~(data['sales_value'] < 5)]
    
    # ������ ������� ������� ������
    data = data[~(data['sales_value'] > 100)]
    
    return data

def postfilter_items(user_id, recommednations):
    pass


def get_similar_items_recommendation(user, model, user_item_matrix, N=5):
    """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""
    user_id = itemid_to_id[user]
    
    top_n_items = user_item_matrix.toarray()[user_id, :]
    top_n_items.sort()
    top_n_items = np.argsort(-top_n_items)
    top_n_items = top_n_items[:N]
    print(top_n_items)
    
    similar_items = []
    for item in top_n_items:
        similar_item = [val[0] for val in model.similar_items(itemid=item, N=2) if val[0] != item]
        similar_items.append(similar_item)
    similar_items = np.array(similar_items).flatten()
    
    return similar_items

def get_similar_users_recommendation(user, model, user_item_matrix, N=5):
    """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""
    
    similar_users = model.similar_users(userid=userid_to_id[user], N=N)
    similar_items = []
    
    for user in similar_users:
        user = user[0]
        top_n_items = user_item_matrix.toarray()[user, :]
        top_n_items = np.argsort(-top_n_items)[:N]
        print(top_n_items)
        similar_items.append(top_n_items)
    similar_items = list(set(np.array(similar_items).flatten()))
    
    return np.random.choice(similar_items, N)