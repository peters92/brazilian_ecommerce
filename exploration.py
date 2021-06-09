#%%
import os
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from tabulate import tabulate

data_files = os.listdir('data/raw/')

data = {}
for file in data_files:
    data[file.split('olist_')[-1].split('.csv')[0]] = \
        pd.read_csv('data/raw/' + file)

# %%
customers_pd = data['customers_dataset']
geolocation_pd = data['geolocation_dataset']
orders_pd = data['orders_dataset']
order_items_pd = data['order_items_dataset']
order_payments_pd = data['order_payments_dataset']
order_reviews_pd = data['order_reviews_dataset']
products_pd = data['products_dataset']
sellers_pd = data['sellers_dataset']
product_category_translation_pd = data['product_category_name_translation']


products_pd = \
    products_pd.merge(product_category_translation_pd, on='product_category_name',
                      how='left')\
               .drop(columns='product_category_name')\
               .rename(columns={'product_category_name_english':\
                                'product_category_name'})

# %%
# plt.plot(geolocation_pd['geolocation_lng'], geolocation_pd['geolocation_lat'],
#          marker=',', linestyle='None')

# %%
# Check for average number of orders per unique customer:
# Is there enough history for customers to make personalized offers?
orders_pd = orders_pd.merge(customers_pd[['customer_id', 'customer_unique_id']],
                            on='customer_id', how='outer')
#%%
customer_groupby = orders_pd.groupby('customer_unique_id')\
                            .agg('count')\
                            .sort_values(by='order_id', ascending=False)

customer_groupby[customer_groupby['order_id']>1].shape

# %%
ax = sns.histplot(data=customer_groupby[customer_groupby['order_id']>1],
                   x='order_id')
ax.set(xlabel='Number of orders per customer',
       title='Histogram of orders per customer')
ax.set_xlim([2,8])
ax.get_figure().savefig('results/visuals/histogram_orders_per_customer.svg')
# %%
# Get customers with exactly 2 purchases, and check if those have any 
# correlation to each other
customers_2_purchase = \
    customer_groupby[customer_groupby['order_id']==2].index.values
# %%
# Next, get the customer_id -> order_id -> product_id -> product_category_name
customers_id_2p = \
    customers_pd[customers_pd['customer_unique_id'].isin(customers_2_purchase)]

orders_2p = customers_id_2p.merge(orders_pd[['order_id', 'customer_id']],
                                  on='customer_id',
                                  how='left')

products_2p = orders_2p.merge(order_items_pd[['order_id', 'product_id']],
                              on='order_id',
                              how='left')

categories_2p = \
    products_2p.merge(products_pd[['product_id', 'product_category_name']],
                      on='product_id',
                      how='left')
categories_2p['product_category_name'] = \
    categories_2p['product_category_name'].fillna('unknown')                      
# %%
unique_categories = categories_2p[['customer_unique_id', 'product_category_name']]\
    .groupby('customer_unique_id').agg(lambda x: ', '.join(x))
# %%
unique_categories = unique_categories['product_category_name']\
                        .apply(lambda x: 'order_1: ' + x.split(', ')[0] + \
                                         ', order_2: ' + x.split(', ')[-1])
#%%

purchase_categories = \
    unique_categories.value_counts()[unique_categories.value_counts() > 1]
# %%
print(tabulate(purchase_categories.to_frame()))


#%%
# Test building predictor table
customers = base['customer_unique_id'].unique()

predictor_dict = {'product_category': [],
                  'review_score': [],
                  'next_product_category': []}
for customer in customers:
    data = base[base['customer_unique_id']==customer]
    predictor_dict['product_category']\
        .append(data.iloc[0]['product_category_name'])
    predictor_dict['review_score'].append(data.iloc[0]['review_score'])
    predictor_dict['next_product_category']\
        .append(data.iloc[1]['product_category_name'])

predictor_table = pd.DataFrame(predictor_dict)
# %%
# Data normalization, one hot encoding etc.
# Data splitting into train, test, validation
# Modelling
# Evaluation

from sklearn.preprocessing import (OneHotEncoder, StandardScaler, LabelEncoder)
# %%
scaler = StandardScaler()
standardized_data = scaler.fit_transform(tmp[['review_score']].values)
#%%
oh_encoder = OneHotEncoder()
oh_data = oh_encoder.fit_transform(tmp[['product_category_name']].values)
# %%
label_encoder = LabelEncoder()
targets = label_encoder.fit_transform(tmp['next_product_category'].values)
# %%
np.concatenate((standardized_data, oh_data.todense(), targets.reshape(-1,1)), axis=1)
# %%
# Geolocation KMeans plots
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5)
clusters = kmeans.fit_predict(geolocation_pd[['geolocation_lat', 'geolocation_lng']])
# %%
geolocation_pd['location_cluster'] = clusters
# %%
sns.scatterplot(data=geolocation_pd.sample(n=1000, random_state=42), x='geolocation_lng', y='geolocation_lat',
                hue='geolocation_state')

# %%
