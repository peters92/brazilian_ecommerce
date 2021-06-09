import numpy
import pandas as pd
import numpy as np
from sklearn.preprocessing import (OneHotEncoder, StandardScaler, LabelEncoder)

def gather_customers_with_n_purchase(data, n):
    # Returns list of customers with exactly 'n' purchase
    data = data.groupby('customer_unique_id').agg('count')
    data = data[data['customer_id']==n]

    return data.index.values

def filter_complete_data_by_customer(customers, data, n):
    data = data[data['customer_unique_id'].isin(customers)]

    # Filter out cases where the customer bought the same product
    # multiple times in the same order
    data = data.drop_duplicates(subset=['customer_unique_id',
                                        'order_id',
                                        'product_id'])

    # Now filter once again people with less than 'n' purchase
    # Then sort by timestamp
    customers = gather_customers_with_n_purchase(data, n=n)
    data = data[data['customer_unique_id'].isin(customers)]
    data = data.sort_values(by=['customer_unique_id',
                                'order_purchase_timestamp'])
    return data

def build_predictor_table(data, columns):   
    customers = data['customer_unique_id'].unique()

    predictor_dict = {column: [] for column in columns}
    for customer in customers:
        customer_data = data[data['customer_unique_id']==customer]

        for column in columns:
            # Note: for the target category we take the next purchase in time
            # e.g. customer_data.iloc[1] as the data is sorted by time
            if column == 'next_product_category':
                predictor_dict[column]\
                    .append(customer_data.iloc[1]['product_category_name'])
            else:
                predictor_dict[column].append(customer_data.iloc[0][column])
        
    predictor_table = pd.DataFrame(predictor_dict)
    return predictor_table

def prepare_modelling_data(data, columns_dict):
    # Encode and normalize categorical variables
    # Encode target labels
    scaler = StandardScaler()
    standardized_data = \
        scaler.fit_transform(data[columns_dict['numerical']].values)

    oh_encoder = OneHotEncoder(handle_unknown='ignore')
    oh_data = oh_encoder.fit_transform(data[columns_dict['categorical']].values)

    label_encoder = LabelEncoder()
    targets = label_encoder.fit_transform(data[columns_dict['target']].values)

    data = np.concatenate((standardized_data,
                           oh_data.todense(),
                           targets.reshape(-1, 1)), axis=1)
    return data, scaler, oh_encoder, label_encoder

def prepare_unlabeled_data(data, columns_dict, encoders):
    # Encode and normalize categorical variables
    # Encode target labels
    scaler = encoders['standard_scaler']
    oh_encoder = encoders['one_hot_encoder']

    standardized_data = scaler.transform(data[columns_dict['numerical']].values)
    oh_data = oh_encoder.transform(data[columns_dict['categorical']].values)

    data = np.asarray(np.concatenate((standardized_data,
                                      oh_data.todense()), axis=1))
    return data

def join_coords(data, geolocation):
    data = _join_coordinates(data, geolocation, new_column_prefix='customer',
                             left_on='customer_zip_code_prefix')
    data = _join_coordinates(data, geolocation, new_column_prefix='seller',
                            left_on='seller_zip_code_prefix')
    
    return data

def _join_coordinates(data, geolocation, left_on, new_column_prefix,
                      right_on='geolocation_zip_code_prefix'):
    geolocation = geolocation[[right_on, 'geolocation_lat', 'geolocation_lng']]
    data = data.merge(geolocation, left_on=left_on, right_on=right_on, 
                      how='left')
    
    for column in ['geolocation_lat', 'geolocation_lng']:
        data.rename(columns={column: new_column_prefix + '_' + column},
                    inplace=True)

    return data

def prepare_product_recommendation_tables(data):
    # Runs processing for different averages/counts for product recommendation
    # then saves the result.
    avg_score = prepare_avg_score_per_product(data)

    product_recommendation_data_dict = {'avg_score_per_product': avg_score}

    return product_recommendation_data_dict

def prepare_avg_score_per_product(data):
    # Calculate the average review score for each product id
    avg_score = data.groupby('product_id').agg('mean')[['review_score']]
    avg_score['product_id'] = avg_score.index
    return avg_score

def prepare_orders_by_sellers(data):
    # Get number of orders per seller
    count_order_by_seller = data.groupby('seller_id').agg('count')['order_id']
    pass