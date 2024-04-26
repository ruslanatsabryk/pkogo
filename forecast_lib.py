import os
import pickle
import csv
import numpy as np

from sklearn.linear_model import OrthogonalMatchingPursuit, OrthogonalMatchingPursuitCV
from sklearn.pipeline import make_pipeline


# #### Функция чтения набора данных
def get_river_dataset(fname, pr_list=None, y_name='H_max'):
    pr_arr = []
    y_arr = []
    with open(fname, newline='') as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            pr_arr_row = []
            for pr in pr_list:
                # Если пустая строка, присвоить None (== np.nan после преобразования во float в np-массиве)
                row_pr = None if len(row[pr]) == 0 else row[pr]
                pr_arr_row.append(row_pr)
            pr_arr.append(pr_arr_row)
            y_arr.append(row[y_name] if row[y_name] else 0)
    X = np.asarray(pr_arr, dtype=np.float64)
    y = np.asarray(y_arr, dtype=np.float64)
    return X, y


# #### Функция формирования тестового набора данных с подстановкой нормированных значений
def test_norm(x, pr_list, norms):
    x_norm = np.copy(x)
    # print('norms')
    # print(norms)
    for col, pr in enumerate(pr_list):
        # print(f'Predictor: {pr}')
        if pr in norms:
            # print(f'x_norm[:, col]: {pr} norm: {norms[pr]}')
            # print(x_norm[:, col])
            ix = (np.isnan(x_norm[:, col]))
            # print(ix)
            x_norm[ix, col] = norms[pr] 
            # print('CORRECTED', x_norm[:, col])
            # x_norm[:, col:col+1] = norms[pr]
    return x_norm


def write_dataset_csv(year, dataset, dataset_name, fieldnames, pr_group=None, mode='training'):
    if mode == 'estimation':
        dir_path = f'results/Estimation/{year}/{dataset_name}/group-{pr_group}/'
        file_name = f'{dataset_name}-гр{pr_group}-Оценка.csv'
    elif mode == 'training':
        dir_path = f'results/Models/{year}/'
        file_name = f'{dataset_name}-гр{pr_group}-Обучение.csv'
    elif mode == 'forecast':
        dir_path = f'results/Forecast/{year}/'
        if pr_group is None:
            file_name = f'{dataset_name}-Прогноз.csv'
        else:
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


# ### Функция прогнозирования
def forecast(year, norms=False, groups_files=False):

    fieldnames = [
        'Group',
        'Prediction',
        'Criterion', 
        'Correlation', 
        'Pm',
        'R2',        
        'R2_t',
        'Method', 
        'Predictors', 
        'Equations',        
    ]
    
    predict_data_dir = f'data/{year}/Predict'
    
    models_dir = f'results/Models/{year}'

    # Получить список файлов .ipnb из results\Models\<year>
    file_list = tuple(filter(lambda fn: '.pickle' in fn, os.listdir(models_dir)))
    # print(file_list)

    # Сделать множество из кортежей (Название-датасета, группа), преобразовать в список, отсортировать
    dataset_group = set()
    for fn in file_list:
        dataset, year, group, model = fn.split('_')
        dataset_group |= {(dataset, group.split('гр')[1])}
    ds_list = sorted(dataset_group)
    # print(ds_list)

    # Для каждого элемента списка создать result_list[one_model_raw: dict]
    
    for ds, group in ds_list:
        # print(f'ds: {ds}, group: {group}')
        result_list = []
        # ds_models = filter(lambda fn: ds in fn and f'гр{group}' in fn, file_list)
        ds_models = filter(lambda fn: ds in fn, file_list)
        # print(list(ds_models))
        # print('------------------------------------------------------------------------')
        # print()
        for file_name in ds_models:
            with open(f'results/Models/{year}/{file_name}', 'rb') as f:
                model_info = pickle.load(f)
                model = model_info['Model_full']
                #model = model_info['Model_train']

                # print(model_info['Group'])
                
                # Прочитать новые признаки (предикторы) для прогноза
                X_new, y = get_river_dataset(f'{predict_data_dir}/{ds}.csv', pr_list=model_info['Predictors_list'])

                # Подстановка нормированных значений для целей тестирования
                if norms:
                    # Подстановка норм в исходный набор признаков
                    X_new = test_norm(X_new, model_info['Predictors_list'], model_info['Norms_data'])

                # print(model_info['Dataset_name'])
                # print(model_info['Predictors_list'])
                # print(model_info['Norms_data'])
                # print(model_info['Method'])
                # print("y")
                # print(y)
                # print(f'R2={model_info["R2"]}, R2_t={model_info["R2_t"]}')
                
                y_predicted = np.ravel(model.predict(X_new))
                # print("y_predicted")
                # print(y_predicted)
                # print('X_new')
                # print(X_new)
                # print(model)
                model_info['Prediction'] = round(y_predicted[-1])
                result_list.append(model_info)
                # Сортировка результатов по каждому датасету
                result_list.sort(key=lambda row: (row['Criterion'], -row['Correlation'], -row['Pm']))
                
        # Запись в .csv файл
        if groups_files:
            write_dataset_csv(year, result_list, ds, fieldnames, pr_group=group, mode='forecast')
        else:
            write_dataset_csv(year, result_list, ds, fieldnames, pr_group=None, mode='forecast')
        print(model_info['Dataset_name'])
        # print('------------------------------------------------------------------------------')
    
    
# #### Запуск процесса прогнозирования
#forecast(2024, norms=True, groups_files=False)





