import pandas as pd
import numpy as np

# Для работы с матрицами
from scipy.sparse import csr_matrix

# Матричная факторизация
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender  # нужен для одного трюка
from implicit.nearest_neighbours import bm25_weight, tfidf_weight


class MainRecommender:
    def __init__(self, user_item_data, item_features, weighting=True):
        
        self.item_features = item_features
        self.user_item_matrix = self.prepare_matrix(user_item_data, self.item_features)
        self.prepare_dicts(self.user_item_matrix)
        
        if weighting:
            self.user_item_matrix = bm25_weight(self.user_item_matrix.T).T 
        
        self.model = self.fit(self.user_item_matrix)
        self.own_recommender = self.fit_own_recommender(self.user_item_matrix)
     
    def prepare_matrix(self, data, item_features):
        data = self.prefilter_items(data, item_features)
        
        user_item_matrix = pd.pivot_table(data, 
                                  index='user_id', columns='item_id', 
                                  values='quantity', # Можно пробоват ьдругие варианты
                                  aggfunc='count', 
                                  fill_value=0
                                 )

        user_item_matrix = user_item_matrix.astype(float) # необходимый тип матрицы для implicit


        return user_item_matrix
    
    def prepare_dicts(self, user_item_matrix):
        """Подготавливает вспомогательные словари"""
        
        userids = user_item_matrix.index.values
        itemids = user_item_matrix.columns.values

        matrix_userids = np.arange(len(userids))
        matrix_itemids = np.arange(len(itemids))

        self.id_to_itemid = dict(zip(matrix_itemids, itemids))
        self.id_to_userid = dict(zip(matrix_userids, userids))

        self.itemid_to_id = dict(zip(itemids, matrix_itemids))
        self.userid_to_id = dict(zip(userids, matrix_userids))
     
    @staticmethod
    def fit_own_recommender(user_item_matrix):
        """Обучает модель, которая рекомендует товары, среди товаров, купленных юзером"""
    
        own_recommender = ItemItemRecommender(K=1, num_threads=4)
        own_recommender.fit(csr_matrix(user_item_matrix).T.tocsr())
        
        return own_recommender
    
    def fit(self, user_item_matrix, n_factors=20, regularization=0.001, iterations=15, num_threads=4):
        """Обучает ALS"""
        model = AlternatingLeastSquares(factors=n_factors, 
                                             regularization=regularization,
                                             iterations=iterations,  
                                             num_threads=num_threads)
        model.fit(csr_matrix(self.user_item_matrix).T.tocsr())
        
        return model
    
    def prefilter_items(self, data, item_features):
        data = data.merge(item_features, how='left', on='item_id', suffixes=('', 'f'))

        # Убираем самые популярные товары.
        popularity = data.groupby('item_id')['user_id'].nunique().reset_index()
        popularity['user_id'] = popularity['user_id']/ data['user_id'].nunique()

        popularity.rename(columns={'user_id': 'share_unique_users'}, inplace=True)
        popularity.sort_values('share_unique_users', inplace=True)

        top_popular = popularity[popularity['share_unique_users'] > 0.5].item_id.tolist()
        data = data[~data['item_id'].isin(top_popular)]

        # Убираем самые непопулярыне товары.
        top_notpopular = popularity[popularity['share_unique_users'] < 0.01].item_id.tolist()
        data = data[~data['item_id'].isin(top_notpopular)]

        # Убараем товары, по которым не было продаж больше 12 месяцев.
        data = data[~(data['week_no'] > data['week_no'].max() - 12 * 4)]

        # Убираем товары из неинтересных категорий (department).
        deps_to_delete = ['GROCERY', 'MISC. TRANS.', 'PASTRY', 'DRUG GM', 'MEAT-PCKGD',
           'SEAFOOD-PCKGD', 'PRODUCE', 'NUTRITION', 'DELI', 'COSMETICS']
        data = data[~(data['department'].isin(deps_to_delete))]

        # Убираем самые дешевые товары. 
        data = data[~(data['sales_value'] < 5)]

        # Убираем самые дорогие товары.
        data = data[~(data['sales_value'] > 100)]
        return data


    def get_similar_items_recommendation(self, user, N=5):
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""
        
        model = self.model
        user_item_matrix = self.user_item_matrix
        
        user_id = self.itemid_to_id[user]

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

        assert len(similar_items) == N, 'Количество рекомендаций != {}'.format(N)
        return similar_items

    def get_similar_users_recommendation(self, user, N=5):
        """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""
        
        model = self.model
        user_item_matrix = self.user_item_matrix
    
        similar_users = model.similar_users(userid=self.userid_to_id[user], N=N)
        similar_items = []

        for user in similar_users:
            user = user[0]
            top_n_items = user_item_matrix.toarray()[user, :]
            top_n_items = np.argsort(-top_n_items)[:N]
            print(top_n_items)
            similar_items.append(top_n_items)
        similar_items = list(set(np.array(similar_items).flatten()))

        return np.random.choice(similar_items, N)