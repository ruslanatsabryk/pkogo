
def get_river_dataset(fname, pr_list=None, y_name='H_max'):
    ### Функция чтения набора данных
    pr_arr = []
    y_arr = []
    with open(fname, newline='') as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            pr_arr_row = []
            for pr in pr_list:
                pr_arr_row.append(row[pr])
            pr_arr.append(pr_arr_row)
            y_arr.append(row[y_name] if row[y_name] else 0)
    X = np.asarray(pr_arr, dtype=np.float64)
    y = np.asarray(y_arr, dtype=np.float64)
    return X, y


def get_s(h_max, h_forecast):
    ### Среднеквадратическая погрешность прогноза S
    n = h_max.shape[0]
    sqr_diff = np.sum((h_max - h_forecast) ** 2) / (n - 1)
    std = sqr_diff ** 0.5
    return std


def get_sigma(h_max):
    ### Среднеквадратическая погрешность климатическая.
    ### (Среднеквадратическое отклонение sigma)
    # Рассчитывается только по всей совокупности данных.
    return np.std(h_max, ddof=1)


def get_delta_dop(sigma):
    ### Допустимая погрешность прогноза delta_dop
    return 0.674 * sigma
    
    
def get_criterion(s, sigma):
    ### Критерий критерий применимости и качества методики S/sigma
    return s / sigma


def get_pm(h_max, h_forecast, delta_dop):
    ### Обеспеченность метода (оправдываемость) Pm
    diff = np.abs(h_max - h_forecast) / delta_dop
    trusted_values = diff[diff <= 1.0]
    m = trusted_values.shape[0]
    n = h_max.shape[0]
    return m / n * 100.00


def get_correlation_ratio(criterion):
    ### Корреляционное отношение
    c_1 = (1 - criterion ** 2)
    ro = c_1 ** 0.5 if c_1 > 0 else 0
    return ro


def write_dataset_csv(year, dataset, dataset_name, fieldnames, pr_group, mode='training'):
    ### Функция записи списка моделей с их характеристиками в csv файл
    if mode == 'estimation':
        dir_path = f'results/Estimation/{year}/{dataset_name}/group-{pr_group}/'
        file_name = f'{dataset_name}-гр{pr_group}-Оценка.csv'
    elif mode == 'training':
        dir_path = f'results/Models/{year}/'
        file_name = f'{dataset_name}-гр{pr_group}-Обучение.csv'
    elif mode == 'forecast':
        dir_path = f'results/Forecast/{year}/'
        file_name = f'{dataset_name}-гр{pr_group}-Прогноз.csv'
    else:
        ...
    
    with open(
        f'{dir_path}'
        f'{file_name}', 
        'w', newline='', encoding='utf-8'
    ) as csvfile:
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, 
                                delimiter=';', extrasaction='ignore')
        writer.writeheader()
        writer.writerows(dataset)


def train_test_split(X, y, n_test, split=True):
    ### Функция разделения набора данных на тренировочный и тестовый
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


def shuffle_xy(X, y, shuffle=True):
    ### Функция перемешивания данных
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


def test_norm(x, pr_list, norms):
    ### Функция формирования тестового набора данных с подстановкой нормированных значений
    x_norm = np.copy(x)
    for col, pr in enumerate(pr_list):
        if pr in norms:
            x_norm[:, col:col+1] = norms[pr]
    return x_norm


def get_datasets():
    ### Функция получения датасетов
    datasets = {
        'Неман-Белица': 'Неман',
        'Неман-Гродно': 'Неман',
        'Неман-Мосты': 'Неман',
        'Неман-Столбцы': 'Неман',

        'Вилия-Стешицы': 'Вилия',
        'Вилия-Михалишки': 'Вилия',

        'ЗападнаяДвина-Сураж': 'ЗападнаяДвина-А',
        'ЗападнаяДвина-Верхнедвинск': 'ЗападнаяДвина-А',
        'ЗападнаяДвина-Витебск': 'ЗападнаяДвина-Б',
        'ЗападнаяДвина-Полоцк': 'ЗападнаяДвина-Б',
    }
    return datasets


def get_predictors(dataset_name, group=None):
    ### Функция получения списка предикторов по названию датасета

    datasets = get_datasets()   
    predictors_lists = {
        'Неман': (
            ['S_2802', 'Smax', 'H_2802', 'X', 'X1', 'X2', 'X3', 'Xs'],
            ['Smax', 'H_2802', 'X', 'X1', 'X3'],
            ['S_2802', 'H_2802', 'X2', 'X3', 'Xs'],
        ),
        'Вилия': (
            ['S_2802', 'Smax', 'H_2802', 'X', 'X1', 'X2', 'X3', 'Xs', 'L_max', 'L_2802', 'Q12', 'Q01', 'Q02', 'Y_sum'],
            ['Smax', 'H_2802', 'X', 'X1', 'X3', 'L_max', 'Y_sum'],
            ['S_2802', 'H_2802', 'X2', 'X3', 'Xs', 'L_2802', 'Y_sum'],
        ),

        'ЗападнаяДвина-А': (
            ['S_2802', 'Smax', 'H_2802', 'X', 'X1', 'X2', 'Xs'],
            ['Smax', 'H_2802', 'X', 'X1'],
            ['S_2802', 'H_2802', 'X2', 'Xs'],
        ),
        'ЗападнаяДвина-Б': (
            ['S_2802', 'Smax', 'H_2802', 'X', 'X1', 'X2', 'Xs', 'Q12', 'Q01', 'Q02', 'Y_sum'],
            ['Smax', 'H_2802', 'X', 'X1', 'Y_sum'],
            ['S_2802', 'H_2802', 'X2', 'Xs', 'Y_sum'],
        ),
    }
    result = predictors_lists[datasets[dataset_name]] if group is None else \
             predictors_lists[datasets[dataset_name]][group]
    return result


def get_norms(dataset_name):
    ### Функция получения нормированных значений предикторов
    norms_list = {
        'Неман-Белица': {'Smax': 59.89, 'X':96.16, 'X1': 46.0, 'X2':35.0},
        'Неман-Гродно': {'Smax': 51.70, 'X':80.27, 'X1': 36.0, 'X2':26.0},
        'Неман-Мосты': {'Smax': 53.62, 'X':88.51, 'X1': 40.0, 'X2':31.0},
        'Неман-Столбцы': {'Smax': 73.68, 'X':101.68, 'X1': 43.0, 'X2':34.0},

        'Вилия-Стешицы': {'Smax': 67.0, 'X': 112.0, 'X1': 40.0, 'X2': 33.0, 'L_max': 60.0},
        'Вилия-Михалишки': {'Smax': 60.0, 'X': 116.0, 'X1': 46.0, 'X2': 37.0, 'L_max': 57.0},

        'ЗападнаяДвина-Сураж': {'S_max': 89.0, 'X':126.0, 'X1': 50.0, 'X2':43.0},
        'ЗападнаяДвина-Верхнедвинск': {'S_max': 62.0, 'X':122.0, 'X1': 68.0, 'X2':56.0},
        'ЗападнаяДвина-Витебск': {'S_max': 80.0, 'X': 134.0, 'X1': 65.0, 'X2': 53.0, 'Y_sum': 46.4},
        'ЗападнаяДвина-Полоцк': {'S_max': 66.0, 'X': 122.0, 'X1': 61.0, 'X2': 53.0, 'Y_sum': 46.5},
    }
    return norms_list[dataset_name]


def augment_data(x_data, y_data, aug_n, aug_pow=2, mirror=True, s=None):
    ### Функция получения аугментированных данных
    #print(x_data)
    data_len = len(y_data)
    
    x_points = np.linspace(0, data_len, data_len)
    
    x_splitted = np.hsplit(x_data, x_data.shape[1])
    #print(x_splitted)

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
    
    # my = np.in1d(y_aug_round, y_data_round)
    # x_aug_clear = x_aug_clear[~my]
    # y_aug_clear = y_aug_clear[~my]
    
    print('x_aug_clear.shape', x_aug_clear.shape)
    print('x_augmented.shape', x_augmented.shape)

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
        plt.plot(y_points, y_data, 'o', y_points_n, y_mirror)
        plt.plot(x_points, x_data[:, 0], 'x', x_points_n, x_mirror[:, 0])

    
    plt.plot(y_points, y_data, 'o', y_points_n, y_augmented)
    plt.plot(x_points, x_data[:, 0], 'x', x_points_n, x_augmented[:, 0])
    plt.show()

    
    
    points_aug_clear = np.linspace(0, y_points, len(y_aug_clear))

    
    
    # plt.plot(y_points, y_data, 'o', points_aug_clear, y_aug_clear)
    # plt.plot(x_points, x_data[:, 0], 'x', points_aug_clear, x_aug_clear[:, 0])
    # plt.show()

    x_result[x_result < 0] = 0
    y_result[y_result < 0] = 0
    
    return x_result, y_result
    
    
def augment_data_2(x_data, y_data, aug_mpl, aug_pow=2, mirror=True, s=None):
    ### Функция получения аугментированных данных
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
    
    print('x_aug_clear.shape', x_aug_clear.shape)
    print('x_augmented.shape', x_augmented.shape)

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
        plt.plot(y_points, y_data, 'o', y_points_n, y_mirror)
        plt.plot(x_points, x_data[:, 0], 'x', x_points_n, x_mirror[:, 0])
   
    plt.plot(y_points, y_data, 'o', points_aug_clear, y_aug_clear)
    plt.plot(x_points, x_data[:, 0], 'x', points_aug_clear, x_aug_clear[:, 0])
    plt.show()

    x_result[x_result < 0] = 0
    y_result[y_result < 0] = 0
    
    return x_result, y_result


def get_transformer(transformer, n_samples=10_000):
    ### Функция получения трансформеров входных данных
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


def get_regressors_list():
    ### Функция получения списка моделей регрессоров
    names = [
        'LinearRegression',
        
        # 'Ridge',
        'RidgeCV',
        
        'ElasticNetCV',
        
        'LassoCV',

        'LarsCV',
        
        'Lars1',
        'Lars2',
        'Lars3',
        'Lars4',
        'Lars5',
        'Lars6',
        'Lars7',
        'Lars8',
        'Lars9',
        'Lars10',
        'Lars11',
        'Lars12',
        'Lars13',
        'Lars14',

        'LassoLarsCV',
        
        'OMPCV',
        
        'OMP1',
        'OMP2',
        'OMP3',
        'OMP4',
        'OMP5',
        'OMP6',
        'OMP7',
        'OMP8',
        'OMP9',
        'OMP10',
        'OMP11',
        'OMP12',
        'OMP13',
        'OMP14',
        
        'BayesianRidge',
        # 'BayesianRidgeCV',
        # 'ARDRegression',
        #'ARDRegressionCV',
        # 'SGDRegressor', 
        'PassiveAggressiveRegressor',
        'HuberRegressor',
        # 'HuberRegressorCV',
        # 'TheilSenRegressor',
        # # 'TheilSenRegressorCV',
        'QuantileRegressor',
        # # 'QuantileRegressorCV',
        
        
        'KNeighborsRegressor',
        # 'NuSVR',
        # 'SVR',
        # # # 'MLPRegressor',
        
        'RandomForestRegressor',
        'ExtraTreesRegressor',
        'HistGradientBoostingRegressor',
        'BaggingRegressor',
        'VotingRegressor',
        'StackingRegressorRidge',
        'AdaBoostRegressor',
    ]
    return names


def get_regressors_objects(grid_search=False):
    ### Функция получения объектов моделей регрессии
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
    
    regressors = [
        LinearRegression(),
        
        # Ridge(random_state=0) if not grid_search else \
        # GridSearchCV(
        #     estimator=Ridge(random_state=0), 
        #     param_grid={"alpha": alphas}
        # ),
        
        RidgeCV(),

        ElasticNetCV(random_state=0),
        
        LassoCV(max_iter=10000, n_alphas=300, random_state=0),  
        
        LarsCV(),
        
        Lars(n_nonzero_coefs=1, random_state=0),
        Lars(n_nonzero_coefs=2, random_state=0),
        Lars(n_nonzero_coefs=3, random_state=0),
        Lars(n_nonzero_coefs=4, random_state=0),
        Lars(n_nonzero_coefs=5, random_state=0),
        Lars(n_nonzero_coefs=6, random_state=0),
        Lars(n_nonzero_coefs=7, random_state=0),
        Lars(n_nonzero_coefs=8, random_state=0),
        Lars(n_nonzero_coefs=9, random_state=0),
        Lars(n_nonzero_coefs=10, random_state=0),
        Lars(n_nonzero_coefs=11, random_state=0),
        Lars(n_nonzero_coefs=12, random_state=0),
        Lars(n_nonzero_coefs=13, random_state=0),
        Lars(n_nonzero_coefs=14, random_state=0),

        LassoLarsCV(max_iter=500, max_n_alphas=1000),

        OrthogonalMatchingPursuitCV(n_jobs=-1),
        
        OrthogonalMatchingPursuit(n_nonzero_coefs=1),
        OrthogonalMatchingPursuit(n_nonzero_coefs=2),
        OrthogonalMatchingPursuit(n_nonzero_coefs=3),
        OrthogonalMatchingPursuit(n_nonzero_coefs=4),
        OrthogonalMatchingPursuit(n_nonzero_coefs=5),
        OrthogonalMatchingPursuit(n_nonzero_coefs=6),
        OrthogonalMatchingPursuit(n_nonzero_coefs=7),
        OrthogonalMatchingPursuit(n_nonzero_coefs=8),
        OrthogonalMatchingPursuit(n_nonzero_coefs=9),
        OrthogonalMatchingPursuit(n_nonzero_coefs=10),
        OrthogonalMatchingPursuit(n_nonzero_coefs=11),
        OrthogonalMatchingPursuit(n_nonzero_coefs=12),
        OrthogonalMatchingPursuit(n_nonzero_coefs=13),
        OrthogonalMatchingPursuit(n_nonzero_coefs=14),
        
        BayesianRidge(),
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

        PassiveAggressiveRegressor(random_state=0) if not grid_search else \
        GridSearchCV(
            estimator=PassiveAggressiveRegressor(random_state=0), 
            param_grid={"C": cc}, 
            n_jobs=-1, 
            cv=3
        ),

        HuberRegressor(max_iter=1000),
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

        QuantileRegressor(),
        
        # QuantileRegressor() if not grid_search else \
        # GridSearchCV(
        #     estimator=QuantileRegressor(), 
        #     param_grid={"alpha": q_alphas}, 
        #     n_jobs=-1
        # ),
        
        
        
        KNeighborsRegressor(n_neighbors=10, metric='euclidean'),
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
       
        
        
        RandomForestRegressor(n_estimators=100, criterion='absolute_error', random_state=0),
        
        ExtraTreesRegressor(n_estimators=100, criterion='squared_error', random_state=0),
        
        HistGradientBoostingRegressor(max_iter=100, loss='absolute_error', max_leaf_nodes=None, min_samples_leaf=10, random_state=0),
        
        BaggingRegressor(
            #KNeighborsRegressor(n_neighbors=20, metric='euclidean'),
            estimator=ExtraTreesRegressor(n_estimators=100, criterion='squared_error', random_state=0), 
            max_samples=0.75, max_features=0.75, n_estimators=10, random_state=0
        ),

        VotingRegressor(
            estimators=[
                ('hgbr', HistGradientBoostingRegressor(max_iter=100, loss='absolute_error', max_leaf_nodes=None, min_samples_leaf=10, random_state=0)), 
                ('omp', ExtraTreesRegressor(n_estimators=100, criterion='squared_error', random_state=0)), 
                ('knr', KNeighborsRegressor(n_neighbors=20, metric='euclidean')),
                ('rfr', RandomForestRegressor(n_estimators=100, criterion='absolute_error', random_state=0)),
            ]
        ),


        StackingRegressor( # RidgeCV - final estimator
            estimators=[
                ('knr', KNeighborsRegressor(n_neighbors=10, metric='euclidean')),
                ('rfr', RandomForestRegressor(n_estimators=100, criterion='absolute_error', random_state=0)),
                ('hgbr', HistGradientBoostingRegressor(max_iter=100, loss='absolute_error', max_leaf_nodes=None, min_samples_leaf=10, random_state=0)), 
                ('etr', ExtraTreesRegressor(n_estimators=100, criterion='squared_error', random_state=0)),
                ('omp', OrthogonalMatchingPursuit(n_nonzero_coefs=5)),
            ],
        ),

        AdaBoostRegressor(estimator=KNeighborsRegressor(n_neighbors=5, metric='euclidean'), n_estimators=100, loss='linear', random_state=0),
        
    ]
    return regressors


def train_models(year, pr_group, n_test=None, norms=True, aug_n=0, aug_mpl=30, aug_pow=2, aug_mirror=False, grid_search=False, scaler_x=None, scaler_y=None, shuffle=True, serial=True, top_best=None):
    ### Функция обучения моделей на полном наборе данных с записью на диск
    
    ds_dir = f'data/{year}/Train'
    
    

    datasets = get_datasets()

    fieldnames = [
        'Predictors', 
        'Equations', 
        'Method', 
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
        ds_groups = get_predictors(ds)
        
        # Итерация по группам предикторов
        for group, pr_list in enumerate(ds_groups):

            if pr_group is not None:
                if group != pr_group:
                    continue
        
            result_list = []
            
            X, y = get_river_dataset(f'{ds_dir}/{ds}.csv', pr_list=pr_list)
    
            # Проверочный набор данных (исходный)
            X_prior = X.copy()
            y_prior = y.copy()
            
            # Полный набор данных
            X_full = X.copy()
            y_full = y.copy()

            if aug_mpl > 1:
                # X_full, y_full = augment_data(X_full, y_full, aug_n, aug_pow=aug_pow, mirror=aug_mirror)
                X_full, y_full = augment_data_2(X_full, y_full, aug_mpl, aug_pow=aug_pow, mirror=aug_mirror)
            
            if shuffle:
                X_full, y_full = shuffle_xy(X_full, y_full, shuffle=True)
            
            if n_test:
                X_train, y_train, X_test, y_test = train_test_split(X_full, y_full, n_test, split=True)
    
            # print("SHAPES:")
            # print("X_train.shape, y_train.shape", X_train.shape, y_train.shape)
            # print("X_test.shape, y_test.shape", X_test.shape, y_test.shape)
            
            norms_data = None
            if norms:
                norms_data = get_norms(ds)
                # Подстановка норм в исходный набор данных (пессимистичный сценарий)
                X_prior = test_norm(X_prior, pr_list, norms_data)
                # Подстановка норм в тестовый набор данных
                X_test = test_norm(X_test, pr_list, norms_data)
                # Подстановка норм в полный набор данных не требуется
  
            # print("X_test:")
            # print(X_test)
            # print("X_train:")
            # print(X_train)

            transformer_y = get_transformer(scaler_y, n_samples=y_train.shape[0]) # !!!
            transformer_x = get_transformer(scaler_x, n_samples=y_train.shape[0]) # !!!
            transformer_y_full = get_transformer(scaler_y, n_samples=y_full.shape[0]) # !!!
            transformer_x_full = get_transformer(scaler_x, n_samples=y_full.shape[0]) # !!!
            print(transformer_y)
            print(transformer_x)

            # Список оцениваемых ререссионных моделей !!!!!
            names = get_regressors_list()
            regressors = get_regressors_objects(grid_search=grid_search)
            regressors_full = get_regressors_objects(grid_search=grid_search)
                
            # Итерация по моделям регрессии
            for name, model, model_full in zip(names, regressors, regressors_full):

                # Препроцессинг - трансформация целевых значений y
                if scaler_y: 
                    regressor = TransformedTargetRegressor(regressor=model, transformer=transformer_y)
                    regressor_full = TransformedTargetRegressor(regressor=model_full, transformer=transformer_y_full)
                else:
                    regressor = model
                    regressor_full = model_full
                
                one_model_row = dict()
                print('X_full.shape', X_full.shape)
                print('y_full.shape', y_full.shape)
                print('X_train.shape', X_train.shape)
                print('y_train.shape', y_train.shape)
                print('X_test.shape', X_test.shape)
                print('y_test.shape', y_test.shape)
                # n_samples = min(10000, y_train.shape[0])
                
                
                # Препроцессинг - трансформация признаков X
                regr = make_pipeline(transformer_x, regressor) if transformer_x else regressor
                regr_full = make_pipeline(transformer_x_full, regressor_full) if transformer_x_full else regressor_full
                
                # Обучение на тренировочном наборе
                # regr = regr.fit(X_train, y_train)
                try:
                    regr.fit(X_train, y_train)
                except ValueError as error:
                    print('Ошибка обучения на тренировочном наборе данных:')
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
                        print("ERROR1 START")
                        print(error)
                        print("ERROR1 FINISH")
                    except Exception as error:
                        print("ERROR2 START")
                        print(error)
                        print("ERROR2 FINISH")
                    
                
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
                    print("ERROR3 START")
                    print(error)
                    print("ERROR3 FINISH")
                    one_model_row['Predictors'] = ""
                    one_model_row['Equations'] = ""
    
                # Год прогноза
                one_model_row['Forecast_year'] = year
                    
                # Название датасета
                one_model_row['Dataset_name'] = ds

                # Название датасета
                one_model_row['Predictors_list'] = pr_list

                # Нормы
                one_model_row['Norms_data'] = norms_data
    
                # Группа предикторов
                one_model_row['Group'] = group
                    
                # Название метода
                one_model_row['Method'] = name
                
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
                one_model_row['S'] = s_forecast
                # по тестовому набору:
                s_forecast_t = get_s(y_test, y_predicted_test) #!!!
                one_model_row['S_t'] = s_forecast_t
                
                # Критерий эффективности метода прогнозирования 
                # климатический S/sigma
                # по исходному набору:
                criterion_forecast = get_criterion(s_forecast, sigma) #!!!
                one_model_row['Criterion'] = criterion_forecast
                # по тестовому набору:
                criterion_forecast_t = get_criterion(s_forecast_t, sigma_t) #!!!
                one_model_row['Criterion_t'] = criterion_forecast_t
    
                
                # Корреляционное отношение ro
                # по исходному набору:
                correlation_forecast = get_correlation_ratio(criterion_forecast)
                one_model_row['Correlation'] = correlation_forecast
                # по тестовому набору:
                correlation_forecast_t = get_correlation_ratio(criterion_forecast_t)
                one_model_row['Correlation_t'] = correlation_forecast_t
                
                
                # Коэффициент детерминации R2
                # по исходному набору:
                one_model_row['R2'] = regr.score(X_prior, y_prior)
                # по тестовому набору:
                one_model_row['R2_t'] = regr.score(X_test, y_test)
                
    
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
                one_model_row['S_f'] = s_forecast_f
    
                # по полному набору:
                criterion_forecast_f = get_criterion(s_forecast_f, sigma_f) #!!!
                one_model_row['Criterion_f'] = criterion_forecast_f
               
                # по полному набору:
                correlation_forecast_f = get_correlation_ratio(criterion_forecast_f)
                one_model_row['Correlation_f'] = correlation_forecast_f
                
                # Коэффициент детерминации R2
                # по полному набору:
                one_model_row['R2_f'] = regr_full.score(X_test, y_test)
    
                print(one_model_row)
    
                
                # Добавление результатов модели в результирующий список по датасету
                result_list.append(one_model_row)
    
                # Запись сериализованного объекта {модель, статистика} в файл
                write_model(one_model_row)

                
                #----------------------------------------------------------------------------------------------
                smodel = pickle.dumps(one_model_row)
                # with open(f'results/Models/{year}/Вилия-Стешицы_2024_гр0_OMP7.pickle', 'rb') as f:
                #     model_info = pickle.load(f, encoding="latin1")
                model_info = pickle.loads(smodel)
                model_full = model_info['Model_full']
                # model_train = model_info['Model_train']
                # Прогноз по исходному набору
                pickled_y_predicted_prior = np.ravel(model_full.predict(X_prior))

                print('------------------------------------------------------------------------------')
                print('predicted_prior train model')
                print(y_predicted_prior)
                
                
                print(name)
                print(y_prior)
                print('predicted_prior full model')
                print(pickled_y_predicted_prior)
                print('------------------------------------------------------------------------------')
                #----------------------------------------------------------------------------------------------
                # Конец итерации по модели
    
            # Сортировка результатов по каждому датасету
            result_list.sort(
                key=lambda row: (row['Criterion'], 
                                 -row['Correlation'], 
                                 -row['Pm'])
            )
    
            datasets_result[ds] = result_list
    
            # Запись в .csv файл
            write_dataset_csv(year, result_list, ds, fieldnames, pr_group=group, mode='training')
        
            # Конец итерации по группе
        
        # Конец итерации по датасету
       
    return datasets_result


def write_model(model_row):
    ### Запись сериализованного объекта {модель, статистика} в файл
    dir_path = f'results/Models/{model_row["Forecast_year"]}'
    
    with open(f'{dir_path}/'
              f'{model_row["Dataset_name"]}_'
              f'{model_row["Forecast_year"]}_'
              f'гр{model_row["Group"]}_'
              f'{model_row["Method"]}.pickle', 'wb') as pf:
        pickle.dump(model_row, pf) #, pickle.HIGHEST_PROTOCOL
        
        
### Запуск процесса обучения моделей множественной регрессии
#_ = train_models(2024, pr_group=None, n_test=100, norms=True, aug_n=1000, aug_mpl=30, aug_pow=2, 
#                 aug_mirror=False, grid_search=False, scaler_x=None, scaler_y=None, 
#                 shuffle=True, serial=True)
