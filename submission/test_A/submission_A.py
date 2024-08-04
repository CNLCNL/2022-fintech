from asyncio import FastChildWatcher
import numpy as np 
import pandas as pd 
import os 
os.chdir(os.path.abspath(os.curdir))
from sklearn import ensemble
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

#Common Model Helpers
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


from sklearn.model_selection import StratifiedKFold

from sklearn.utils import shuffle
from sklearn.base import BaseEstimator, RegressorMixin

import featuretools as ft

class PseudoLabeler(BaseEstimator, RegressorMixin):
    # 伪标签
    def __init__(self, model, test, features, target, sample_rate=0.2, seed=42):
        self.sample_rate = sample_rate
        self.seed = seed
        self.model = model
        self.model.seed = seed
        
        self.test = test
        self.features = features
        self.target = target
        
    def get_params(self, deep=True):
        return {
            "sample_rate": self.sample_rate,
            "seed": self.seed,
            "model": self.model,
            "test": self.test,
            "features": self.features,
            "target": self.target
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

        
    def fit(self, X, y):
        if self.sample_rate > 0.0:
            augemented_train = self.__create_augmented_train(X, y)
            self.model.fit(
                augemented_train[self.features],
                augemented_train[self.target]
            )
        else:
            self.model.fit(X, y)
        
        return self

    def __create_augmented_train(self, X, y):
        num_of_samples = int(len(self.test) * self.sample_rate)
        
        # Train the model and creat the pseudo-labels
        self.model.fit(X, y)
        pseudo_labels = self.model.predict(self.test[self.features])
        
        # Add the pseudo-labels to the test set
        augmented_test = self.test.copy(deep=True)
        augmented_test[self.target] = pseudo_labels
        
        # Take a subset of the test set with pseudo-labels and append in onto
        # the training set
        sampled_test = augmented_test.sample(n=num_of_samples)
        temp_train = pd.concat([X, y], axis=1)
        augemented_train = pd.concat([sampled_test, temp_train])

        return shuffle(augemented_train)
        
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def get_model_name(self):
        return self.model.__class__.__name__

def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df

def create_feature_ft():
    train_ft = pd.read_excel('fintech训练营/train.xlsx')
    test_ft = pd.read_excel('fintech训练营/test_A榜.xlsx')
    datasets = [train_ft,test_ft]
    for dataset in datasets:
        for i in dataset.columns:
            dataset[i]=dataset[i].apply(lambda x : np.nan if x=='?' else x)

    ignore = ['CUST_UID','LABEL']
    categorical = ['MON_12_CUST_CNT_PTY_ID',
                'AI_STAR_SCO',
                'WTHR_OPN_ONL_ICO',
                'SHH_BCK',
                'LGP_HLD_CARD_LVL',
                'NB_CTC_HLD_IDV_AIO_CARD_SITU']
    features = [feat for feat in train_ft.columns if feat not in (ignore + categorical)]
    target_feature = 'LABEL'

    es = ft.EntitySet(id='fintech')  # 用id标识实体集
    es=es.add_dataframe(
        dataframe_name = "fintech_train",
        dataframe = train_ft[features],
        index = '1',
        make_index = True
    )
    es=es.add_dataframe(
        dataframe_name = "fintech_test",
        dataframe = test_ft[features],
        index = '2',
        make_index = True
    )

    feature_train, feature_defs_train = ft.dfs(entityset=es, 
                                        target_dataframe_name='fintech_train',
                                        agg_primitives=["mean", "sum", "mode"],
                                        trans_primitives=['add_numeric', 'subtract_numeric', 'multiply_numeric', 'divide_numeric'], # 2列相加减乘除来生成
                                        max_depth = 1)
    feature_test, feature_defs_test = ft.dfs(entityset=es, 
                                        target_dataframe_name='fintech_test',
                                        agg_primitives=["mean", "sum", "mode"],
                                        trans_primitives=['add_numeric', 'subtract_numeric', 'multiply_numeric', 'divide_numeric'], # 2列相加减乘除来生成
                                        max_depth = 1)

    feature_train_last = pd.concat([feature_train,train_ft[categorical+[target_feature]]],axis=1)
    feature_test_last = pd.concat([feature_test,test_ft[categorical]],axis=1)

    label = LabelEncoder()
    datasets=[feature_train_last,feature_test_last]
    for dataset in datasets:
        for i in categorical:            
            dataset[i] = label.fit_transform(dataset[i])

    for dataset in datasets:
        for i in dataset.columns:
            dataset[i] = dataset[i].apply(lambda x : np.nan if x==np.inf else x) # 除会产生inf, xgboost预测需要转换成nan

    feature_train_last = reduce_mem_usage(feature_train_last)
    feature_test_last = reduce_mem_usage(feature_test_last)

    return feature_train_last,feature_test_last

def training(train, test, features, target_feature):
    params_hgbc = {'max_iter': 300, 
            'learning_rate': 0.03, 
            'max_leaf_nodes': 1980, 
            'max_depth': 6, 
            'random_state': 38647831
    }

    params_lgb = {'n_estimators': 300, 
                'learning_rate': 0.03, 
                'num_leaves': 1840, 
                'max_depth': 7, 
                'subsample': 0.7936669434888015, 
                'subsample_freq': 1, 
                'colsample_bytree': 0.4517958053910891,
                'random_state': 28649880
    }

    params_xgb = {'n_estimators': 300, 
                'learning_rate': 0.03, 
                'num_leaves': 1880, 
                'max_depth': 8, 
                'subsample': 0.7574143599011826, 
                'subsample_freq': 1, 
                'colsample_bytree': 0.682578966844618,
                'verbosity':0,
                'random_state': 3178455
    }

    params_cbc = {'iterations': 1000, 
                'learning_rate': 0.03, 
                'depth': 8, 
                'subsample': 0.32454338188635595,
                "random_state": 2022,
                'verbose':0
    }

    vote_est = [
        #Ensemble Methods: http://scikit-learn.org/stable/modules/ensemble.html
        ('hgbc',ensemble.HistGradientBoostingClassifier(**params_hgbc)),
        #lightbgm
        ('lgb', LGBMClassifier(**params_lgb)),
        #xgboost: http://xgboost.readthedocs.io/en/latest/model.html
        ('xgb', XGBClassifier(**params_xgb)),
        ('cbc',CatBoostClassifier(**params_cbc))

    ]

    strtfdKFold = StratifiedKFold(n_splits=5,random_state=100,shuffle=True)
    grid_soft = ensemble.VotingClassifier(estimators = vote_est , voting = 'soft', weights = [0.15,0.35,0.35,0.15])
    X_train=train[features]
    y_train=train[target_feature]
    X_test=test[features]
    kfold = strtfdKFold.split(X_train, y_train)
    pred=pd.DataFrame()
    for k, (train1, test1) in enumerate(kfold):
        pseudo = PseudoLabeler(grid_soft,test,features,target_feature,sample_rate=1)
        pseudo.fit(X_train.iloc[train1,:], y_train.iloc[train1])
        pred_lgb = pseudo.predict_proba(X_test)[:,1]
        pred[str(k)]=pred_lgb
        print(k)

    pred['result']=(pred['0']+pred['1']+pred['2']+pred['3']+pred['4'])/5
    return pred

if __name__=='__main__':
    train, test = create_feature_ft() # featuretools构造特征
    test_original = pd.read_excel('fintech训练营/test_A榜.xlsx')
    important = pd.read_csv('important/no_add_original_400/important168.csv') # 筛选出的特征名
    ignore = ['CUST_UID','LABEL']
    original_feature = list(test_original.columns)
    features = [feat for feat in important['feature_names'] if feat not in ignore]
    target_feature = 'LABEL'
    pred = training(train, test, features, target_feature) # 训练预测
    sub=pd.DataFrame(test_original['CUST_UID'])
    sub['prob']=pred['result']
    with open('test_grid_soft_pseudo_168_weights_cbc.txt','w') as file:
        for i in sub.index:
            file.write(sub.loc[i,'CUST_UID']+' '+str(sub.loc[i,'prob'])+'\n') # 0.95511