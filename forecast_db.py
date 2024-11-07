#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pickle
import csv
import numpy as np
import psycopg2
import psycopg2.extras


# In[2]:


def close_db(connection, cursor):
    if connection:
        cursor.close()
        connection.close()
    


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
    except (Exception, psycopg2.Error) as error:
        print(f"Error connecting to database: {error}")

    return conn, cursor


# #### Функция чтения набора данных

# In[4]:


def get_predict_dataset_db(year, station_id, pr_list=None):
    conn, cursor = connect_db()
    
    predictors = ', '.join(pr_list)
    sql_observations = f"""
    SELECT {predictors} FROM maxlevel.predict_data
    WHERE predict_year = %s and station_id = %s 
    """
    station_arg = (year, station_id,)
    cursor.execute(sql_observations, station_arg)
    observations = cursor.fetchall()
    
    close_db(conn, cursor)
        
    X = np.asarray(observations, dtype=np.float64)
    return X


# #### Функция формирования тестового набора данных с подстановкой нормированных значений

# In[5]:


def test_norm(x, pr_list, norms):
    x_norm = np.copy(x)
    for col, pr in enumerate(pr_list):
        if pr in norms:
            ix = (np.isnan(x_norm[:, col]))
            x_norm[ix, col] = norms[pr] 
    return x_norm


# ### Функция получения списка моделей из БД (таблица models)

# In[6]:


def get_models_db(year):
    conn, cursor = connect_db(dict_cursor=True)
    sql = """
    select m.forecast_year, m.model_id, m.station_id, m.predictors_id, p.predictors, m.model_file 
    from maxlevel.models m
    inner join maxlevel.predictors_groups p on m.predictors_id = p.predictors_id
    where forecast_year = %s
    order by m.station_id, m.model_id, m.group_n, m.predictors_id;
    """
    cursor.execute(sql, (year, ))
    models = cursor.fetchall()
    close_db(conn, cursor)
    
    return models


# ### Функция прогнозирования

# In[7]:


def forecast_db(year, stations_id=None):

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
    
    models = get_models_db(year)
    for model_row in models:
        if stations_id:
            if not model_row['station_id'] in stations_id:
                continue
        model_file = model_row['model_file']
        with open(model_file, 'rb') as f:
            model_info = pickle.load(f)
            model = model_info['Model_full']
            pr_list = model_row['predictors'].replace(' ' , '').split(',')
            X_new = get_predict_dataset_db(year, model_row['station_id'], pr_list=pr_list)
            print(X_new)
            y_predicted = np.ravel(model.predict(X_new))
            model_row['prediction'] = round(y_predicted[-1])
            # print(model_row['Prediction'])
            write_forecast_db(model_row)

    


# ### Функция записи данных в БД

# In[8]:


def write_forecast_db(model_row):
    conn, cursor = connect_db(dict_cursor=True)

    sql = """
    INSERT INTO maxlevel.forecasts (forecast_year, model_id, station_id, predictors_id, h_max)
    VALUES (%s, %s, %s, %s, %s)
    """
    args = (model_row['forecast_year'], model_row['model_id'], model_row['station_id'], model_row['predictors_id'], model_row['prediction'])
    cursor.execute(sql, args)
    conn.commit()

    close_db(conn, cursor)
 


# #### Пример запуска процесса прогнозирования

# In[10]:


# forecast_db(2024, stations_id=[73131, 73111])

