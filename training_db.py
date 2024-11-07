#!/usr/bin/env python
# coding: utf-8

# #### Импорт необходимых объектов

# In[1]:


import os
import csv
import numpy as np
from scipy.interpolate import splrep, splev
import pickle

from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from sklearn.compose import TransformedTargetRegressor

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import Normalizer

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.linear_model import Lars, LarsCV
from sklearn.linear_model import LassoLars, LassoLarsCV
from sklearn.linear_model import OrthogonalMatchingPursuit, OrthogonalMatchingPursuitCV
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import ARDRegression
from sklearn.linear_model import TweedieRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import TheilSenRegressor
from sklearn.linear_model import QuantileRegressor
from sklearn.neural_network import MLPRegressor

from sklearn.linear_model import RidgeCV, LassoCV, LassoLarsCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import NuSVR, SVR

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import AdaBoostRegressor

# import matplotlib.pyplot as plt

import psycopg2


# #### Ошибки климатического/природного прогноза для каждого года delta50

# In[2]:


def get_delta50(h_max, delta_dop, h_max_avg=None, h_max_forecast=None):
    if h_max_forecast is None:
        # delta50 климатическая
        return (h_max - h_max_avg) / delta_dop
    else:
        # delta50 прогноза
        return (h_max - h_max_forecast) / delta_dop
  


# ### БД. Функция подключения к БД ПК ОГО

# In[3]:


def connect_db(dict_cursor=False):
    db_params = {
        "host": "192.168.29.134",
        "port": "5432",
        "database": "pkogo",
        "user": "pkogouser",
        "password": "pkogouser",
    }
    try:
        conn = psycopg2.connect(**db_params)
        if dict_cursor:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        else:
            cursor = conn.cursor()
        # print(cursor)
        # print("DB connection works")
    except (Exception, psycopg2.Error) as error:
        print(f"Error connecting to database: {error}")

    return conn, cursor


# ### БД. Функции закрытия подключения к БД ПК ОГО

# In[4]:


def close_db(connection, cursor):
    if connection:
        cursor.close()
        connection.close()
    


# #### БД. Функция чтения набора данных из БД

# In[5]:


def get_river_dataset_db(station_id, pr_list=None, y_name='H_max'):
    conn, cursor = connect_db()
    
    predictors = ', '.join(pr_list) + f', {y_name}' + ', obs_year'
    sql_observations = f"""
    SELECT {predictors} FROM maxlevel.observations
    WHERE station_id = %s
    ORDER BY obs_year
    """
    station_arg = (station_id, )
    cursor.execute(sql_observations, station_arg)
    observations = cursor.fetchall()
    
    close_db(conn, cursor)
        
    X_y_years = np.asarray(observations, dtype=np.float64)
    X = np.asarray(X_y_years[:, :-2], dtype=np.float64)
    y = np.asarray(X_y_years[:, -2], dtype=np.float64)
    obs_years = np.asarray(X_y_years[:, -1], dtype=np.float64)
    return X, y, obs_years


# In[6]:


# get_river_dataset_db(74021, pr_list=['S_2802', 'Smax', 'H_2802', 'X', 'X1', 'X2', 'X3', 'Xs'])


# #### Сумма, средний, высший, низший уровни

# In[7]:


def get_sum(h_max):
    return np.sum(h_max)
    
def get_avg(h_max):
    return np.mean(h_max)
    
def get_max(h_max):
    return np.amax(h_max)
    
def get_min(h_max):
    return np.amin(h_max)


# #### Среднее значение максимальных уровней воды

# In[8]:


def get_hmax_avg(h_max):
    # Среднее значение h_max.
    # Рассчитывается только по всей совокупности данных.
    return np.mean(h_max)


# #### Среднеквадратическая погрешность прогноза S

# In[9]:


def get_s(h_max, h_forecast):
    # Среднеквадратическая погрешность прогноза
    n = h_max.shape[0]
    sqr_diff = np.sum((h_max - h_forecast) ** 2) / (n - 1)
    std = sqr_diff ** 0.5
    return std    


# #### Среднеквадратическое отклонение sigma

# In[10]:


def get_sigma(h_max):
    # Среднеквадратическая погрешность климатическая.
    # Рассчитывается только по всей совокупности данных.
    return np.std(h_max, ddof=1)


# #### Допустимая погрешность прогноза delta_dop

# In[11]:


def get_delta_dop(sigma):
    return 0.674 * sigma


# #### Критерий критерий применимости и качества методики S/sigma

# In[12]:


def get_criterion(s, sigma):
    return s / sigma


# #### Климатическая обеспеченность Pk

# In[13]:


def get_pk(h_max, h_max_avg, delta_dop):
    diff = np.abs(h_max - h_max_avg) / delta_dop
    trusted_values = diff[diff <= 1.0]
    m = trusted_values.shape[0]
    n = h_max.shape[0]
    return m / n * 100.00


# #### Обеспеченность метода (оправдываемость) Pm

# In[14]:


def get_pm(h_max, h_forecast, delta_dop):
    diff = np.abs(h_max - h_forecast) / delta_dop
    trusted_values = diff[diff <= 1.0]
    m = trusted_values.shape[0]
    n = h_max.shape[0]
    return m / n * 100.00


# #### Корреляционное отношение

# In[15]:


def get_correlation_ratio(criterion):
    c_1 = (1 - criterion ** 2)
    ro = c_1 ** 0.5 if c_1 > 0 else 0
    return ro


# #### Функция записи списка моделей с их характеристиками в csv файл

# In[16]:


# def write_dataset_csv(year, dataset, dataset_name, fieldnames, pr_group, mode='training'):
#     if mode == 'estimation':
#         dir_path = f'results/Estimation/{year}/{dataset_name}/group-{pr_group}/'
#         file_name = f'{dataset_name}-гр{pr_group}-Оценка.csv'
#     elif mode == 'training':
#         dir_path = f'results/Models/{year}/'
#         file_name = f'{dataset_name}-гр{pr_group}-Обучение.csv'
#     elif mode == 'forecast':
#         dir_path = f'results/Forecast/{year}/'
#         file_name = f'{dataset_name}-гр{pr_group}-Прогноз.csv'
#     else:
#         ...
    
#     with open(
#         f'{dir_path}'
#         f'{file_name}', 
#         'w', newline='', encoding='utf-8'
#     ) as csvfile:
        
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames, 
#                                 delimiter=';', extrasaction='ignore')
#         writer.writeheader()
#         writer.writerows(dataset)


# #### Функция разделения набора данных на тренировочный и тестовый

# In[17]:


def train_test_split(X, y, n_test, split=True):
    if split:   
        X_train = X[:-n_test].copy()
        y_train = y[:-n_test].copy()
        X_test = X[-n_test:].copy()
        y_test = y[-n_test:].copy()
    else:
        X_train = X.copy()
        y_train = y.copy()
        X_test = X.copy()
        y_test = y.copy()
    return X_train, y_train, X_test, y_test


# #### Функция перемешивания данных

# In[18]:


def shuffle_xy(X, y, shuffle=True):
    if shuffle:
        # Перемешивание данных
        Xy = np.column_stack((X, y))
        rng = np.random.default_rng(42)
        rng.shuffle(Xy)
        y_sh = Xy[:, -1]
        X_sh = Xy[:,:-1]
    else:
        y_sh = y.copy()
        X_sh = X.copy()
    return X_sh, y_sh


# #### Функция формирования тестового набора данных с подстановкой нормированных значений

# In[19]:


def test_norm(x, pr_list, norms):
    x_norm = np.copy(x)
    for col, pr in enumerate(pr_list):
        if pr in norms:
            x_norm[:, col:col+1] = norms[pr]
    return x_norm


# #### Функция получения словаря гидропостов из БД

# In[20]:


def get_datasets_db():
    conn, cursor = connect_db()
    sql = "SELECT station, station_id FROM maxlevel.stations"
    cursor.execute(sql)
    result = cursor.fetchall()
    close_db(conn, cursor)

    return {station: id for station, id in result}


# In[21]:


# get_datasets_db()


# #### Функция получения списка предикторов по названию гидропоста из БД

# In[22]:


def get_predictors_db(station_id, group=None):
    conn, cursor = connect_db()
    sql = """
    SELECT p.predictors_id, p.predictors
    FROM maxlevel.predictors_groups p
    INNER JOIN maxlevel.stations s on p.method_id = s.method_id 
    WHERE s.station_id = %s
    ORDER BY p.group_n
    """
    cursor.execute(sql, (station_id,))
    groups = cursor.fetchall()
    # print(groups)
    # Преобразование строк в списки
    groups_list = []
    for s in groups:
        sg = s[1].replace(' ', '')
        groups_list.append((s[0], sg.split(',')))
    close_db(conn, cursor)
    
    return groups_list


# In[23]:


# get_predictors_db(74021)


# #### Функция получения нормированных значений предикторов

# In[24]:


def get_norms(dataset_name):
    norms_list = {
        'Белица': {'Smax': 59.89, 'X':96.16, 'X1': 46.0, 'X2':35.0},
        'Гродно': {'Smax': 51.70, 'X':80.27, 'X1': 36.0, 'X2':26.0},
        'Мосты': {'Smax': 53.62, 'X':88.51, 'X1': 40.0, 'X2':31.0},
        'Столбцы': {'Smax': 73.68, 'X':101.68, 'X1': 43.0, 'X2':34.0},

        'Стешицы': {'Smax': 67.0, 'X': 112.0, 'X1': 40.0, 'X2': 33.0, 'L_max': 60.0},
        'Михалишки': {'Smax': 60.0, 'X': 116.0, 'X1': 46.0, 'X2': 37.0, 'L_max': 57.0},

        'Сураж': {'S_max': 89.0, 'X':126.0, 'X1': 50.0, 'X2':43.0},
        'Верхнедвинск': {'S_max': 62.0, 'X':122.0, 'X1': 68.0, 'X2':56.0},
        'Витебск': {'S_max': 80.0, 'X': 134.0, 'X1': 65.0, 'X2': 53.0, 'Y_sum': 46.4},
        'Полоцк': {'S_max': 66.0, 'X': 122.0, 'X1': 61.0, 'X2': 53.0, 'Y_sum': 46.5},
    }
    return norms_list[dataset_name]


# #### Функция получения аугментированных данных

# In[25]:


def augment_data(x_data, y_data, aug_mpl, aug_pow=2, mirror=True, s=None):
    #print(x_data)
    data_len = len(y_data)
    
    x_points = np.linspace(0, data_len, data_len)
    
    x_splitted = np.hsplit(x_data, x_data.shape[1])
    #print(x_splitted)

    aug_n = round(data_len * (aug_mpl - 1) / (data_len - 1)) * (data_len - 1) + data_len
    
    x_list = []
    for arr in x_splitted:
        x_spl = splrep(x_points, arr, k=aug_pow, s=s)
        x_points_n = np.linspace(0, data_len, aug_n)
        x_col_augmented = splev(x_points_n, x_spl)
        x_list.append(x_col_augmented)
    x_augmented = np.array(x_list).T

    y_points = np.linspace(0, data_len, data_len)
    y_spl = splrep(y_points, y_data, k=aug_pow, s=s)
    y_points_n = np.linspace(0, data_len, aug_n)
    y_augmented = splev(y_points_n, y_spl)
    
    x_aug_round = np.round(x_augmented, decimals=-1)
    y_aug_round = np.round(y_augmented, decimals=1)

    x_data_round = np.round(x_data, decimals=-1)
    y_data_round = np.round(y_data, decimals=1)   
    
    
    mx = (x_aug_round[:, None] == x_data_round).all(-1).any(1)
    x_aug_clear = x_augmented[~mx].copy()
    y_aug_clear = y_augmented[~mx].copy()
    points_aug_clear = y_points_n[~mx]
    
    # print('x_aug_clear.shape', x_aug_clear.shape)
    # print('x_augmented.shape', x_augmented.shape)

    if mirror:
        x_mirror = np.mean(x_augmented) - x_augmented + np.mean(x_augmented)
        y_mirror = np.mean(y_augmented) - y_augmented + np.mean(y_augmented)
    
        x_result = np.vstack((x_aug_clear, x_mirror))
        y_result = np.hstack((y_aug_clear, y_mirror))

    else:
        x_result = x_aug_clear
        y_result = y_aug_clear
    
    if mirror:
        ...
    #     plt.plot(y_points, y_data, 'o', y_points_n, y_mirror)
    #     plt.plot(x_points, x_data[:, 0], 'x', x_points_n, x_mirror[:, 0])
   
    # plt.plot(y_points, y_data, 'o', points_aug_clear, y_aug_clear)
    # plt.plot(x_points, x_data[:, 0], 'x', points_aug_clear, x_aug_clear[:, 0])
    # plt.show()

    x_result[x_result < 0] = 0
    y_result[y_result < 0] = 0
    
    return x_result, y_result
    


# #### Функция получения трансформеров входных данных

# In[26]:


def get_transformer(transformer, n_samples=10_000):
    scaler = (
        StandardScaler() if transformer == 'standard' else \
        MinMaxScaler() if transformer == 'minmax' else \
        MaxAbsScaler() if transformer == 'maxabs' else \
        RobustScaler() if transformer == 'robust' else \
        QuantileTransformer(output_distribution='uniform', n_quantiles=n_samples, random_state=0) if transformer == 'uniform' else \
        QuantileTransformer(output_distribution='normal', n_quantiles=n_samples, random_state=0) if transformer == 'normal' else \
        PowerTransformer(method='box-cox', standardize=False) if transformer == 'normal-bc' else \
        PowerTransformer(method='yeo-johnson', standardize=False) if transformer == 'normal-yj' else \
        PowerTransformer(method='box-cox', standardize=True) if transformer == 'normal-bc-st' else \
        PowerTransformer(method='yeo-johnson', standardize=True) if transformer == 'normal-yj-st' else \
        None
    )
    return scaler


# #### Функция получения списка моделей регрессоров из БД

# In[27]:


def get_regressors_list_db():
    conn, cursor = connect_db()

    sql = """
    SELECT algorithm_id, algorithm FROM maxlevel.algorithms
    WHERE active is true
    ORDER BY algorithm_id
    """
    cursor.execute(sql)
    algorithms = cursor.fetchall()
    
    close_db(conn, cursor)

    return algorithms
    


# In[28]:


# get_regressors_list_db()


# #### Функция получения объектов моделей регрессии

# In[29]:


def get_regressors_objects(grid_search=False):
    # Инициализация генератора случайных чисел для
    # для обеспечения воспроизводимости результатов
    rng = np.random.RandomState(0)

    # Наборы гиперпараметров моделей для алгоритма кроссвалидации
    # Гиперпараметры для Ridge, Lasso, ElasticNet, LassoLars, HuberRegressor
    alphas = np.logspace(-4, 3, num=100)
    
    # Гиперпараметры для ElasticNet
    l1_ratio = np.linspace(0.01, 1.0, num=50)
    
    # Гиперпараметры для BayesianRidge
    alphas_init = np.linspace(0.5, 2, 5)
    lambdas_init = np.logspace(-3, 1, num=5)
    
    # Гиперпараметры для ARDRegression
    alphas_lambdas = np.logspace(-7, -4, num=4)
    
    # Гиперпараметры для SGDRegressor
    losses = ['squared_error', 'huber', 
              'epsilon_insensitive', 'squared_epsilon_insensitive']
    sgd_alphas = np.logspace(-4, 1, num=100)
   
    # Гиперпараметры для PassiveAggressiveRegressor
    cc = np.linspace(0.1, 1.5, 50)
    
    # Гиперпараметры для HuberRegressor
    epsilons = np.append(np.linspace(1.1, 2.0, 10), [1.35])
    
    # Гиперпараметры для TheilSenRegressor
    # n_subsamples = np.arange(15, 24)
    n_subsamples = (16, 24, 32)
    
    # Гиперпараметры для QuantileRegressor
    # q_alphas = np.linspace(0, 1, 5)
    q_alphas = (0.1, 1, 2)    
    
    regressors = {
        'LinearRegression': LinearRegression(),
        
        # Ridge(random_state=0) if not grid_search else \
        # GridSearchCV(
        #     estimator=Ridge(random_state=0), 
        #     param_grid={"alpha": alphas}
        # ),
        
        'RidgeCV': RidgeCV(),

        'ElasticNetCV': ElasticNetCV(random_state=0),

        'LassoCV': LassoCV(max_iter=10000, n_alphas=300, random_state=0),  

        'LarsCV': LarsCV(),
        
        'Lars1': Lars(n_nonzero_coefs=1, random_state=0),
        'Lars2': Lars(n_nonzero_coefs=2, random_state=0),
        'Lars3': Lars(n_nonzero_coefs=3, random_state=0),
        'Lars4': Lars(n_nonzero_coefs=4, random_state=0),
        'Lars5': Lars(n_nonzero_coefs=5, random_state=0),
        'Lars6': Lars(n_nonzero_coefs=6, random_state=0),
        'Lars7': Lars(n_nonzero_coefs=7, random_state=0),
        'Lars8': Lars(n_nonzero_coefs=8, random_state=0),
        'Lars9': Lars(n_nonzero_coefs=9, random_state=0),
        'Lars10': Lars(n_nonzero_coefs=10, random_state=0),
        'Lars11': Lars(n_nonzero_coefs=11, random_state=0),
        'Lars12': Lars(n_nonzero_coefs=12, random_state=0),
        'Lars13': Lars(n_nonzero_coefs=13, random_state=0),
        'Lars14': Lars(n_nonzero_coefs=14, random_state=0),

        'LassoLarsCV': LassoLarsCV(max_iter=500, max_n_alphas=1000),

        'OMPCV': OrthogonalMatchingPursuitCV(n_jobs=-1),
        
        'OMP1': OrthogonalMatchingPursuit(n_nonzero_coefs=1),
        'OMP2': OrthogonalMatchingPursuit(n_nonzero_coefs=2),
        'OMP3': OrthogonalMatchingPursuit(n_nonzero_coefs=3),
        'OMP4': OrthogonalMatchingPursuit(n_nonzero_coefs=4),
        'OMP5': OrthogonalMatchingPursuit(n_nonzero_coefs=5),
        'OMP6': OrthogonalMatchingPursuit(n_nonzero_coefs=6),
        'OMP7': OrthogonalMatchingPursuit(n_nonzero_coefs=7),
        'OMP8': OrthogonalMatchingPursuit(n_nonzero_coefs=8),
        'OMP9': OrthogonalMatchingPursuit(n_nonzero_coefs=9),
        'OMP10': OrthogonalMatchingPursuit(n_nonzero_coefs=10),
        'OMP11': OrthogonalMatchingPursuit(n_nonzero_coefs=11),
        'OMP12': OrthogonalMatchingPursuit(n_nonzero_coefs=12),
        'OMP13': OrthogonalMatchingPursuit(n_nonzero_coefs=13),
        'OMP14': OrthogonalMatchingPursuit(n_nonzero_coefs=14),
        
        'BayesianRidge': BayesianRidge(),
        # BayesianRidge() if not grid_search else \
        # GridSearchCV(
        #     estimator=BayesianRidge(),
        #     param_grid={"alpha_init": alphas_init, "lambda_init": lambdas_init}, 
        #     n_jobs=-1
        # ),

        # ARDRegression(),
        
        # ARDRegression() if not grid_search else \
        # GridSearchCV(
        #     estimator=ARDRegression(), 
        #     param_grid={"alpha_1": alphas_lambdas, "alpha_2": alphas_lambdas,
        #                 "lambda_1": alphas_lambdas,"lambda_2": alphas_lambdas}, 
        #     n_jobs=-1
        # ),

        # SGDRegressor(random_state=0) if not grid_search else \
        # GridSearchCV(
        #     estimator=SGDRegressor(random_state=0), 
        #     param_grid={"loss": losses, "alpha": sgd_alphas}, 
        #     n_jobs=-1
        # ),

        'PassiveAggressiveRegressor': PassiveAggressiveRegressor(random_state=0) if not grid_search else \
        GridSearchCV(
            estimator=PassiveAggressiveRegressor(random_state=0), 
            param_grid={"C": cc}, 
            n_jobs=-1, 
            cv=3
        ),

        'HuberRegressor': HuberRegressor(max_iter=1000),
        # HuberRegressor(max_iter=1000) if not grid_search else \
        # GridSearchCV(
        #     estimator=HuberRegressor(), 
        #     param_grid={"epsilon": epsilons, "alpha": alphas}, 
        #     n_jobs=-1 
        # ),

        # TheilSenRegressor(random_state=0, n_jobs=-1),
        # TheilSenRegressor(random_state=0, n_jobs=-1) if not grid_search else \
        # GridSearchCV(
        #     estimator=TheilSenRegressor(random_state=0, n_jobs=-1), 
        #     param_grid={"n_subsamples": n_subsamples}, 
        #     n_jobs=-1
        # ),

        'QuantileRegressor': QuantileRegressor(),
        
        # QuantileRegressor() if not grid_search else \
        # GridSearchCV(
        #     estimator=QuantileRegressor(), 
        #     param_grid={"alpha": q_alphas}, 
        #     n_jobs=-1
        # ),
        
        
        
        'KNeighborsRegressor': KNeighborsRegressor(n_neighbors=10, metric='euclidean'),
        # NuSVR(C=5.0, nu=0.9, kernel='poly', degree=3),
        # SVR(C=5.0, epsilon=0.2, kernel='poly', degree=3),
        
        
        # MLPRegressor(
        #     hidden_layer_sizes=(3, ), 
        #     activation='identity', 
        #     max_iter=100000, 
        #     early_stopping=True, 
        #     learning_rate='constant',
        #     learning_rate_init=0.00025,
        #     batch_size=75,
        #     solver='adam',
        #     random_state=0
        # ),
       
        
        
        'RandomForestRegressor': RandomForestRegressor(n_estimators=100, criterion='absolute_error', random_state=0),
        
        'ExtraTreesRegressor': ExtraTreesRegressor(n_estimators=100, criterion='squared_error', random_state=0),
        
        'HistGradientBoostingRegressor':HistGradientBoostingRegressor(max_iter=100, loss='absolute_error', max_leaf_nodes=None, min_samples_leaf=10, random_state=0),
        
        'BaggingRegressor': BaggingRegressor(
            #KNeighborsRegressor(n_neighbors=20, metric='euclidean'),
            estimator=ExtraTreesRegressor(n_estimators=100, criterion='squared_error', random_state=0), 
            max_samples=0.75, max_features=0.75, n_estimators=10, random_state=0
        ),

        'VotingRegressor': VotingRegressor(
            estimators=[
                ('hgbr', HistGradientBoostingRegressor(max_iter=100, loss='absolute_error', max_leaf_nodes=None, min_samples_leaf=10, random_state=0)), 
                ('omp', ExtraTreesRegressor(n_estimators=100, criterion='squared_error', random_state=0)), 
                ('knr', KNeighborsRegressor(n_neighbors=20, metric='euclidean')),
                ('rfr', RandomForestRegressor(n_estimators=100, criterion='absolute_error', random_state=0)),
            ]
        ),


        'StackingRegressorRidgeCV': StackingRegressor( # RidgeCV - final estimator
            estimators=[
                ('knr', KNeighborsRegressor(n_neighbors=10, metric='euclidean')),
                ('rfr', RandomForestRegressor(n_estimators=100, criterion='absolute_error', random_state=0)),
                ('hgbr', HistGradientBoostingRegressor(max_iter=100, loss='absolute_error', max_leaf_nodes=None, min_samples_leaf=10, random_state=0)), 
                ('etr', ExtraTreesRegressor(n_estimators=100, criterion='squared_error', random_state=0)),
                ('omp', OrthogonalMatchingPursuit(n_nonzero_coefs=5)),
            ],
        ),

        'AdaBoostRegressor': AdaBoostRegressor(estimator=KNeighborsRegressor(n_neighbors=5, metric='euclidean'), n_estimators=100, loss='linear', random_state=0),
        
    }
    return regressors


# #### Функция обучения моделей на полном наборе данных с записью на диск и в БД (таблица models)

# In[30]:


def train_models(year, pr_group, n_test=None,
                 norms=True, aug_n=0, aug_mpl=30,
                 aug_pow=2, aug_mirror=False, grid_search=False,
                 scaler_x=None, scaler_y=None, shuffle=True,
                 serial=True, top_best=None,
                 stations_id=None):
    
    # ds_dir = f'data/{year}/Train'
    
    # {station: id, ...}
    if stations_id:
        datasets = {st: stid for st, stid in get_datasets_db().items() if stid in stations_id}
    else:
        datasets = get_datasets_db()
                     
    fieldnames = [
        'Predictors', 
        'Equations', 
        'Algorithm', 
        'Criterion', 
        'Correlation', 
        'Pm',
        'R2',

        'Criterion_t', 
        'Correlation_t', 
        'Pm_t',
        'R2_t',

        'Criterion_f', 
        'Correlation_f', 
        'Pm_f',
        'R2_f',

        # 'Group',
        # 'Augmentation',
        # 'Data size',
        # 'Normalization',
        # 'Equations',
    ]

    # Описание структуры данных переменной datasets_result
    # datasets_result = {
    #     "hydropost_0": [
    #         { model_row }
    #         { model_row }
    #     ],
    #     ...,
    #     "hydropost_n": [
    #         { model_row }
    #         { model_row }
    #     ],
    # }
    
    
    # Итерация по датасетам
    datasets_result = dict()
    for ds in datasets:

        # Получить все группы по датасету
        ds_groups = get_predictors_db(datasets[ds])
        
        # Итерация по группам предикторов
        for group, (predictors_id, pr_list) in enumerate(ds_groups):
            if group == 0:
                # По группе 0 модели не обучаются - это все предикторы + год + максимальный уровень и его дата
                continue
            if pr_group is not None:
                if group != pr_group:
                    continue
        
            result_list = []
            
            X, y, obs_years = get_river_dataset_db(datasets[ds], pr_list=pr_list)
    
            # Проверочный набор данных (исходный)
            X_prior = X.copy()
            y_prior = y.copy()
            
            # Полный набор данных
            X_full = X.copy()
            y_full = y.copy()

            if aug_mpl > 1:
                X_full, y_full = augment_data(X_full, y_full, aug_mpl, aug_pow=aug_pow, mirror=aug_mirror)
            
            if shuffle:
                X_full, y_full = shuffle_xy(X_full, y_full, shuffle=True)
            
            if n_test:
                X_train, y_train, X_test, y_test = train_test_split(X_full, y_full, n_test, split=True)
            
            norms_data = None
            if norms:
                norms_data = get_norms(ds) # сделать для БД
                # Подстановка норм в исходный набор данных (пессимистичный сценарий)
                X_prior = test_norm(X_prior, pr_list, norms_data)
                # Подстановка норм в тестовый набор данных
                X_test = test_norm(X_test, pr_list, norms_data)
                # Подстановка норм в полный набор данных не требуется
  
            transformer_y = get_transformer(scaler_y, n_samples=y_train.shape[0]) # !!!
            transformer_x = get_transformer(scaler_x, n_samples=y_train.shape[0]) # !!!
            transformer_y_full = get_transformer(scaler_y, n_samples=y_full.shape[0]) # !!!
            transformer_x_full = get_transformer(scaler_x, n_samples=y_full.shape[0]) # !!!

            # Список оцениваемых ререссионных моделей !!!!!
            algorithms = get_regressors_list_db()
            
            regressors = get_regressors_objects(grid_search=grid_search)
            regressors_full = get_regressors_objects(grid_search=grid_search)
                
            # Итерация по моделям регрессии
            for alg_id, alg in algorithms:
                model = regressors[alg]
                model_full = regressors_full[alg]

                # Препроцессинг - трансформация целевых значений y
                if scaler_y: 
                    regressor = TransformedTargetRegressor(regressor=model, transformer=transformer_y)
                    regressor_full = TransformedTargetRegressor(regressor=model_full, transformer=transformer_y_full)
                else:
                    regressor = model
                    regressor_full = model_full
                
                one_model_row = dict()                
                
                # Препроцессинг - трансформация признаков X
                regr = make_pipeline(transformer_x, regressor) if transformer_x else regressor
                regr_full = make_pipeline(transformer_x_full, regressor_full) if transformer_x_full else regressor_full
                
                # Обучение на тренировочном наборе
                # regr = regr.fit(X_train, y_train)
                try:
                    regr.fit(X_train, y_train)
                except ValueError as error:
                    print(f'{ds}, гр. {group}, {alg} - Ошибка обучения на тренировочном наборе данных:')
                    print(error)
                    continue
                if serial:
                    serial_model = pickle.dumps(regr)
                    regr = pickle.loads(serial_model)
                
                # Обученная на тренировочном наборе данных модель
                one_model_row['Model_train'] = regr
                    
                # Прогноз по исходному набору на тренировочной модели
                y_predicted_prior = np.ravel(regr.predict(X_prior))
                
                # Прогноз по тестовому набору на тренировочной модели 
                y_predicted_test = np.ravel(regr.predict(X_test))
    
                
                # Очистка значений строк предикторов и уравнений перед переходом к следующей модели
                coef = None
                intercept = None
                
                try:
                    coef = regr.best_estimator_.coef_
                    intercept = regr.best_estimator_.intercept_
                    
                    if isinstance(intercept, np.ndarray):
                        intercept = intercept[0]
                except Exception as error:
                                    
                    try:
                        coef = regr.coef_
                        intercept = regr.intercept_
                    
                        if isinstance(intercept, np.ndarray):
                            intercept = intercept[0]
                        # print("ERROR1 START")
                        # print(error)
                        # print("ERROR1 FINISH")
                    except Exception as error:
                        ...
                        # print("ERROR2 START")
                        # print(error)
                        # print("ERROR2 FINISH")
                    
                
                try:
                    # Коэффициенты уравнения (если есть)
                    coef = np.around(np.ravel(coef), 3)
                    intercept = round(intercept, 3)
                    
                    predictors_coef = {f: c for f, c 
                                       in zip(pr_list, coef) if c != 0.0}
                    
                    predictors = ", ".join(predictors_coef.keys())
                    
                    equation = (
                        str(intercept) 
                        + ' ' 
                        + ' '.join(str(c) + '*' 
                                   + f for f, c in predictors_coef.items())
                    )
                    
                    equation = equation.replace(" -", "-")
                    equation = equation.replace(" ", " + ")
                    equation = equation.replace("-", " - ")
        
                    one_model_row['Predictors'] = predictors
                    one_model_row['Equations'] = equation
                except Exception as error:
                    # print("ERROR3 START")
                    # print(error)
                    # print("ERROR3 FINISH")
                    one_model_row['Predictors'] = ", ".join(pr_list)
                    one_model_row['Equations'] = "Непараметрическая (нелинейная) модель"
    
                
                
                # Код алгоритма
                one_model_row['Algorithm_id'] = alg_id

                # Код гидропоста
                one_model_row['Station_id'] = datasets[ds]

                # Код списка предикторов
                one_model_row['Predictors_id'] = predictors_id
                    
                # Год прогноза
                one_model_row['Forecast_year'] = year

                # Название датасета (гидропоста)
                one_model_row['Dataset_name'] = ds

                # Исходный список предикторов
                one_model_row['Predictors_list'] = pr_list

                # Нормы
                one_model_row['Norms_data'] = norms_data
    
                # Группа предикторов
                one_model_row['Group'] = group
                    
                # Название метода
                one_model_row['Algorithm'] = alg

                # Путь к файлу модели на бэкенде
                model_file = (
                    f'results/Models/{one_model_row["Forecast_year"]}/'
                    f'{one_model_row["Dataset_name"]}_'
                    f'{one_model_row["Forecast_year"]}_'
                    f'гр{one_model_row["Group"]}_'
                    f'{one_model_row["Algorithm"]}.pickle'
                )
                one_model_row['Model_file'] = model_file

                # Сумма, максимум, минимум, среднее максимальных уровней
                # по исходному набору:
                one_model_row['H_sum'] = get_sum(y_prior)
                one_model_row['H_max'] = get_max(y_prior)
                one_model_row['H_min'] = get_min(y_prior)
                h_max_avg = get_hmax_avg(y_prior)
                one_model_row['H_avg'] = h_max_avg
                # по тестовому набору:
                if n_test:
                    one_model_row['H_sum_t'] = get_sum(y_test)
                    one_model_row['H_max_t'] = get_max(y_test)
                    one_model_row['H_min_t'] = get_min(y_test)
                    h_max_avg_t = get_hmax_avg(y_test)
                    one_model_row['H_avg_t'] = h_max_avg_t
                
                # Среднеквадратическое отклонение
                # по исходному набору:
                sigma = get_sigma(y_prior) #!!!
                one_model_row['Sigma'] = sigma
                # по тестовому набору:
                sigma_t = get_sigma(y_test) #!!!
                one_model_row['Sigma_t'] = sigma_t
                
                # Допустимая погрешность прогноза
                # по исходному набору:
                delta_dop = get_delta_dop(sigma) #!!!
                one_model_row['Delta_dop'] = delta_dop
                # по тестовому набору:
                delta_dop_t = get_delta_dop(sigma_t) #!!!
                one_model_row['Delta_dop_t'] = delta_dop_t

                # Обеспеченность климатическая Pk 
                # по исходному набору:
                pk = get_pk(y_prior, h_max_avg, delta_dop)
                one_model_row['Pk'] = pk
                # по тестовому набору:
                pk_t = get_pk(y_test, h_max_avg_t, delta_dop_t)
                one_model_row['Pk_t'] = pk_t
    
                # Обеспеченность метода (оправдываемость) Pm
                # по исходному набору:
                pm = get_pm(y_prior, y_predicted_prior, delta_dop) #!!!
                one_model_row['Pm'] = pm
                # по тестовому набору:
                pm_t = get_pm(y_test, y_predicted_test, delta_dop_t) #!!!
                one_model_row['Pm_t'] = pm_t
    
                # Среднеквадратическая погрешность прогноза
                # по исходному набору:
                s_forecast = get_s(y_prior, y_predicted_prior) #!!!
                one_model_row['S'] = s_forecast if s_forecast < 10e8 else 99_999_999
                # по тестовому набору:
                s_forecast_t = get_s(y_test, y_predicted_test) #!!!
                one_model_row['S_t'] = s_forecast_t if s_forecast_t < 10e8 else 99_999_999
                
                # Критерий эффективности метода прогнозирования 
                # климатический S/sigma
                # по исходному набору:
                criterion_forecast = get_criterion(s_forecast, sigma) #!!!
                # print('!!!!!!!!!!!------criterion_forecast------!!!!!!!!!!!!!')
                # print(criterion_forecast)
                one_model_row['Criterion'] = criterion_forecast if criterion_forecast < 10e6 else 999_999
                criterion_sqr = criterion_forecast ** 2.0
                one_model_row['Criterion_sqr'] = criterion_sqr
                
                # по тестовому набору:
                criterion_forecast_t = get_criterion(s_forecast_t, sigma_t) #!!!
                one_model_row['Criterion_t'] = criterion_forecast_t if criterion_forecast_t < 10e6 else 999_999
                criterion_sqr_t = criterion_forecast_t ** 2.0
                one_model_row['Criterion_sqr_t'] = criterion_sqr_t
                
                # Корреляционное отношение ro
                # по исходному набору:
                correlation_forecast = get_correlation_ratio(criterion_forecast)
                one_model_row['Correlation'] = correlation_forecast
                # print('!!!!!!!!!!!------correlation_forecast------!!!!!!!!!!!!!')
                # print(correlation_forecast)
                # по тестовому набору:
                correlation_forecast_t = get_correlation_ratio(criterion_forecast_t)
                one_model_row['Correlation_t'] = correlation_forecast_t
                
                
                # Коэффициент детерминации R2
                # по исходному набору:
                r2 = regr.score(X_prior, y_prior)
                one_model_row['R2'] = r2 if r2 > -10e6 else -999_999
                # по тестовому набору:
                r2_t = regr.score(X_test, y_test)
                one_model_row['R2_t'] = r2_t if r2_t > -10e6 else -999_999
                
    
                # Обучение на полном наборе данных
                try:
                    regr_full = regr_full.fit(X_full, y_full)
                except ValueError as error:
                    print('Ошибка обучения на полном наборе данных:')
                    print(error)
                    continue
                if serial:
                    serial_model_full = pickle.dumps(regr_full)
                    regr_full = pickle.loads(serial_model_full)

                # Обученная на полных данных модель
                one_model_row['Model_full'] = regr_full
    
                # Прогноз по полному набору (производится на "тестовых" данных) 
                y_predicted_full = np.ravel(regr_full.predict(X_test))         
    
                # по полному набору:
                sigma_f = get_sigma(y_test) #!!!
                one_model_row['Sigma_f'] = sigma_f
    
                # по полному набору:
                delta_dop_f = get_delta_dop(sigma_f) #!!!
                one_model_row['Delta_dop_f'] = delta_dop_f
    
                # по полному набору:
                pm_f = get_pm(y_test, y_predicted_full, delta_dop_f) #!!!
                one_model_row['Pm_f'] = pm_f
    
                # по полному набору:
                s_forecast_f = get_s(y_test, y_predicted_full) #!!!
                one_model_row['S_f'] = s_forecast_f if s_forecast_f < 10e8 else 99_999_999
    
                # по полному набору:
                criterion_forecast_f = get_criterion(s_forecast_f, sigma_f) #!!!
                one_model_row['Criterion_f'] = criterion_forecast_f if criterion_forecast_f < 10e6 else 999_999
               
                # по полному набору:
                correlation_forecast_f = get_correlation_ratio(criterion_forecast_f)
                one_model_row['Correlation_f'] = correlation_forecast_f
                
                # Коэффициент детерминации R2
                # по полному набору:
                r2_f = regr_full.score(X_test, y_test)
                one_model_row['R2_f'] = r2_f if r2_f > -10e6 else -999_999
    
                # print(one_model_row)
    
                
                # Добавление результатов модели в результирующий список по датасету
                result_list.append(one_model_row)
    
                # Запись сериализованного объекта {модель, статистика} в файл
                write_model(one_model_row)
                # Запись данных модели в БД (таблица models)
                model_id = write_model_db(one_model_row)

                # Создание проверочного прогноза и запись в БД (таблица test_forecasts)
                # X_prior передается уже с нормами
                verify_forecast(model_id, model_info=one_model_row, xy=(X_prior, y_prior), obs_years=obs_years)

                
                #----------------------------------------------------------------------------------------------
                smodel = pickle.dumps(one_model_row)
                # with open(f'results/Models/{year}/Вилия-Стешицы_2024_гр0_OMP7.pickle', 'rb') as f:
                #     model_info = pickle.load(f, encoding="latin1")
                model_info = pickle.loads(smodel)
                model_full = model_info['Model_full']
                # model_train = model_info['Model_train']
                # Прогноз по исходному набору
                pickled_y_predicted_prior = np.ravel(model_full.predict(X_prior))

                # Конец итерации по модели

                # print(f'Рассчитано: {ds}, {alg}, гр: {group}')
    
            # Сортировка результатов по каждому датасету
            result_list.sort(
                key=lambda row: (row['Criterion'], 
                                 -row['Correlation'], 
                                 -row['Pm'])
            )
    
            datasets_result[ds] = result_list
    
            # Запись в .csv файл
            # write_dataset_csv(year, result_list, ds, fieldnames, pr_group=group, mode='training')
        
            # Конец итерации по группе
        
        # Конец итерации по датасету
       
    return datasets_result


# ### Запись сериализованного объекта {модель, статистика} в файл

# In[31]:


def write_model(model_row):
    current_dir = os.getcwd()
    dir_path = f'{current_dir}/results/Models/{model_row["Forecast_year"]}'
    
    with open(f'{dir_path}/'
              f'{model_row["Dataset_name"]}_'
              f'{model_row["Forecast_year"]}_'
              f'гр{model_row["Group"]}_'
              f'{model_row["Algorithm"]}.pickle', 'wb') as pf:
        pickle.dump(model_row, pf) #, pickle.HIGHEST_PROTOCOL
        


# ### Запись данных о модели в БД

# In[32]:


def write_model_db(model_row):
    conn, cursor = connect_db()

    # Удалить данные модели из таблицы models
    sql_delete = """
    DELETE FROM maxlevel.models
    WHERE algorithm_id = %s
    AND   station_id = %s
    AND   predictors_id = %s
    AND   forecast_year = %s
    """
    arg_delete = (
        model_row['Algorithm_id'],
        model_row['Station_id'],
        model_row['Predictors_id'],
        model_row['Forecast_year'],
    )
    cursor.execute(sql_delete, arg_delete)
    conn.commit()

    # Записать данные модели в таблицу models
    sql_insert = """
    INSERT INTO 
    maxlevel.models (
    algorithm_id, station_id, predictors_id, forecast_year,
    model_file, group_n, predictors, equations, dataset_name,
    algorithm,

    sigma,
    sigma_t,
    sigma_f,
    delta_dop,
    delta_dop_t,
    delta_dop_f,
    pm,
    pm_t,
    pm_f,
    s,
    s_t,
    s_f,
    criterion,
    criterion_t,
    criterion_f,
    correlation,
    correlation_t,
    correlation_f,
    r2,
    r2_t,
    r2_f,

    pk,
    criterion_sqr,
    h_sum,
    h_avg,
    h_max,
    h_min
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s)
    """
    arg_insert = (
        model_row['Algorithm_id'],
        model_row['Station_id'],
        model_row['Predictors_id'],
        model_row['Forecast_year'],
        model_row['Model_file'],
        model_row['Group'],
        model_row['Predictors'],
        model_row['Equations'],
        model_row['Dataset_name'],
        model_row['Algorithm'],

        model_row['Sigma'],
        model_row['Sigma_t'],
        model_row['Sigma_f'],
        model_row['Delta_dop'],
        model_row['Delta_dop_t'],
        model_row['Delta_dop_f'],
        model_row['Pm'],
        model_row['Pm_t'],
        model_row['Pm_f'],
        model_row['S'],
        model_row['S_t'],
        model_row['S_f'],
        model_row['Criterion'],
        model_row['Criterion_t'],
        model_row['Criterion_f'],
        model_row['Correlation'],
        model_row['Correlation_t'],
        model_row['Correlation_f'],
        model_row['R2'],
        model_row['R2_t'],
        model_row['R2_f'],

        model_row['Pk'],
        model_row['Criterion_sqr'],
        model_row['H_sum'],
        model_row['H_avg'],
        model_row['H_max'],
        model_row['H_min'],
        
        
    )
    cursor.execute(sql_insert, arg_insert)
    conn.commit()

    # Получить model_id записанной модели
    sql_model_id = """
    SELECT model_id FROM maxlevel.models
    WHERE algorithm_id=%s and station_id=%s and predictors_id=%s and forecast_year=%s
    """
    arg_model_id = (
        model_row['Algorithm_id'],
        model_row['Station_id'],
        model_row['Predictors_id'],
        model_row['Forecast_year']
    )
    cursor.execute(sql_model_id, arg_model_id)
    model_id = cursor.fetchone()[0]
    close_db(conn, cursor)

    return model_id


# ### Функция вычисления проверочных прогнозов

# In[33]:


def verify_forecast(model_id, model_info, xy, obs_years):
    
    # ds_dir = f"data/{model_info['Forecast_year']}/Train"
    
    fieldnames = [
        '№', 
        'Год',
        'Hmax фактический', 
        'Hф-Hср', 
        '(Hф-Hср)^2', 
        
        'δ50% Погрешность климатических прогнозов '
        'в долях от допустимой погрешности',
        
        'Hmax прогнозный', 
        'Hф-Hп', 
        '(Hф-Hп)^2', 
        
        'δ50% Погрешность проверочных прогнозов '
        'в долях от допустимой погрешности',
    ]

    X, y = xy
    X_test = X.copy()
    y_test = y.copy()
    
    # Forecast
    h_max_forecast = np.ravel(model_info['Model_full'].predict(X_test))

    # print('verify_forecast')
    
    # Hсредний
    h_max_avg = np.mean(y)

    # H - Hсредний
    diff_fact = y_test - h_max_avg

    # (H - Hсредний) в квадрате
    diff_fact_sqr = diff_fact ** 2

    # Погрешность климатических прогнозов в долях от допустимой погрешности
    delta_dop = get_delta_dop(get_sigma(y))
    error_climate = get_delta50(y_test, delta_dop, h_max_avg=h_max_avg)

    # H - Hпрогнозный
    diff_forecast = y_test - h_max_forecast

    # (H - Hпрогнозный) в квадрате
    diff_forecast_sqr = diff_forecast ** 2       

    # Погрешность проверочных прогнозов в долях от допустимой погрешности
    error_forecast = get_delta50(
        y_test, delta_dop, h_max_forecast=h_max_forecast
    )

    # Номер по порядку
    rows_num = y_test.shape[0]
    npp = np.arange(1, rows_num + 1, 1)

    # Конкатенация массивов
    att_tuple = (
        npp, 
        obs_years, 
        y_test, 
        diff_fact, 
        diff_fact_sqr, 
        error_climate, 
        h_max_forecast, 
        diff_forecast, 
        diff_forecast_sqr, 
        error_forecast
    )
    
    arr = np.column_stack(att_tuple)
    arr_db = arr.copy()
    arr = arr.tolist()    
    arr_db[:, 0] = model_id
    
    # Обеспеченность метода (оправдываемость) Pm
    pm = get_pm(y_test, h_max_forecast, delta_dop)

    # Запись проверочного прогноза в БД
    write_test_forecast_db(model_id, values=arr_db)
    
    # # Запись проверочного прогноза в csv файл
    # with open(
    #     f"results/Estimation/{model_info['Forecast_year']}/{model_info['Dataset_name']}/group-{model_info['Group']}/{model_info['Dataset_name']}"
    #     f"-проверочный-гр{model_info['Group']}-{model_id}.csv", 
    #     'w', 
    #     newline='', 
    #     encoding='utf-8'
    # ) as csvfile:
        
    #     stat_header = (
    #         f"Таблица  - "
    #         f"Проверочные прогнозы максимумов весеннего половодья\n"
    #         f"р.{model_info['Dataset_name']}\n"
    #         f"Предикторы:;; {model_info['Predictors']}\n"
    #         f"Уравнение:;; {model_info['Equations']}\n"
    #         f"Модель:;; {model_info['Algorithm']}\n\n"
    #     )
        
    #     csvfile.write(stat_header)
    #     writer = csv.writer(csvfile, delimiter=';')
    #     writer.writerow(fieldnames)
    #     writer.writerows(arr)
      
    #     stat_footer = (
    #         f"Сумма;;{model_info['H_sum']}\n"  
    #         f"Средний;;{model_info['H_avg']}\n" 
    #         f"Высший;;{model_info['H_max']}\n"
    #         f"Низший;;{model_info['H_min']}\n\n"
            
    #         f"σ = ;;{model_info['Sigma']};;σ -;"
    #         f"среднеквадратическое отклонение (см)\n" 
            
    #         f"δдоп =;;{model_info['Delta_dop']};;δдоп -;"
    #         f"допустимая погрешность прогноза (см)\n" 
            
    #         f"Pк =;;{model_info['Pk']};;Pк -;"
    #         f"климатическая обеспеченность в %\n"
            
    #         f"Pм =;;{model_info['Pm']};;Pм -;"
    #         f"обеспеченность метода в %\n"
            
    #         f"S =;;{model_info['S']};;;"
    #         f"(допустимой погрешности проверочных прогнозов)\n"
            
    #         f"S/σ =;;{model_info['Criterion']};;S -;"
    #         f"среднеквадратическая погрешность (см)\n" 
            
    #         f"(S/σ)^2 =;;{model_info['Criterion_sqr']};;S/σ -;"
    #         f"критерий эффективности метода прогнозирования\n"
            
    #         f"ρ =;;{model_info['Correlation']};;ρ -;"
    #         f"корреляционное отношение\n"
            
    #         f";;;;;(оценка эффективности метода прогнозирования)\n"
    #         f";;;;δ50% -;погрешность (ошибка) прогнозов (см)\n"
    #     )
        
    #     csvfile.write(stat_footer) 


# #### Функция записи проверочного прогноза в БД

# In[34]:


def write_test_forecast_db(model_id, values):
    conn, cursor = connect_db()

    sql = """
    INSERT INTO maxlevel.test_forecasts (
        model_id, 
        obs_year, 
        h_max_fact, 
        diff_fact, 
        diff_fact_sqr,
        error_climate,
        h_max_forecast, 
        diff_forecast, 
        diff_forecast_sqr, 
        error_forecast
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    cursor.executemany(sql, values)
    conn.commit()

    close_db(conn, cursor)


# ### Функция удаления моделей из БД

# In[35]:


def delete_models_db(stations_id=None):
    conn, cursor = connect_db()

    if stations_id:
        # Удалить модели для указанных гидропостов из таблицы models
        sql = """
        DELETE FROM maxlevel.models
        WHERE station_id = ANY(%s)
        """
        arg = (stations_id,)
        cursor.execute(sql, arg)
    else:
        # Удалить все модели из таблицы models
        sql = """
        DELETE FROM maxlevel.models
        """
        cursor.execute(sql)
    conn.commit()
    close_db(conn, cursor)


# #### Запуск процесса удаления моделей множественной регрессии

# In[38]:


# delete_models_db(stations_id=[])


# #### Запуск процесса обучения моделей множественной регрессии

# In[40]:


_ = train_models(2024, pr_group=None, n_test=100, norms=True, aug_n=1000, aug_mpl=30, aug_pow=2, 
                 aug_mirror=False, grid_search=False, scaler_x=None, scaler_y=None, 
                 shuffle=True, serial=True,
                 stations_id=[73131, 73111])


# In[ ]:




