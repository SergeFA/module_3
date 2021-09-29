#!/usr/bin/env python
# coding: utf-8

# # Загрузка Pandas и очистка данных

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv('main_task.csv')


# In[3]:


df.info()


# In[4]:


df.head()


# In[5]:


df['Price Range'] = df['Price Range'].fillna(0)


# In[ ]:





# In[ ]:


# Ваш код по очистке данных и генерации новых признаков
# При необходимости добавьте ячейки


# In[ ]:





# In[6]:


df = df.drop(['City', 'Cuisine Style', 'Price Range', 'Reviews', 'URL_TA', 'ID_TA'], axis=1)


# In[7]:


df['Number of Reviews'] = df['Number of Reviews'].fillna('0')


# In[8]:


df


# In[9]:


df.info()


# # Разбиваем датафрейм на части, необходимые для обучения и тестирования модели

# In[10]:


# Х - данные с информацией о ресторанах, у - целевая переменная (рейтинги ресторанов)
X = df.drop(['Restaurant_id', 'Rating'], axis = 1)
y = df['Rating']


# In[11]:


# Загружаем специальный инструмент для разбивки:
from sklearn.model_selection import train_test_split


# In[12]:


# Наборы данных с меткой "train" будут использоваться для обучения модели, "test" - для тестирования.
# Для тестирования мы будем использовать 25% от исходного датасета.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


# # Создаём, обучаем и тестируем модель

# In[13]:


# Импортируем необходимые библиотеки:
from sklearn.ensemble import RandomForestRegressor # инструмент для создания и обучения модели
from sklearn import metrics # инструменты для оценки точности модели


# In[14]:


# Создаём модель
regr = RandomForestRegressor(n_estimators=100)

# Обучаем модель на тестовом наборе данных
regr.fit(X_train, y_train)

# Используем обученную модель для предсказания рейтинга ресторанов в тестовой выборке.
# Предсказанные значения записываем в переменную y_pred
y_pred = regr.predict(X_test)


# In[15]:


# Сравниваем предсказанные значения (y_pred) с реальными (y_test), и смотрим насколько они в среднем отличаются
# Метрика называется Mean Absolute Error (MAE) и показывает среднее отклонение предсказанных значений от фактических.
print('MAE:', metrics.mean_absolute_error(y_test, y_pred))


# In[ ]:




