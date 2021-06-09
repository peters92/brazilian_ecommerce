import pandas as pd
pd.options.mode.chained_assignment = None #TODO: Fix class with 1 sample

def get_recommended_product(prediction_row, 
                            complete_data,
                            rec_tables,
                            product_number,
                            estimation_n):
    # Returns recommended products and other data for one customer at a time,
    # Based on their next predicted product category
    # Filter out the customer from the data
    complete_data = \
        complete_data[complete_data['customer_unique_id']!= \
                      prediction_row['customer_unique_id']]
    avg_score_per_product = rec_tables['avg_score_per_product']

    predicted_category = prediction_row['predicted_category']

    # Get all products in category
    # TODO: change this part to select products based on location cluster as well
    category_products = get_products_in_category(predicted_category,
                                                 complete_data)
    # Get best product from top n by average review score
    top_n_products = get_top_n_products_by_score(category_products,
                                                 avg_score_per_product,
                                                 product_number,
                                                 estimation_n)
    
    product_scores = top_n_products['review_score']
    products = top_n_products['product_id']

    expected_values = get_expected_values(prediction_row, 
                                          complete_data, 
                                          products, n=estimation_n)
    
    recommendation = {'product_scores': product_scores, 'products': products,
                      'expected_values': expected_values}

    return recommendation

def get_products_in_category(category, complete_data):
    products = \
        complete_data[complete_data['product_category_name']==category]
    
    return products.product_id.unique()

def get_top_n_products_by_score(category_products, avg_score_per_product,
                                product_number, estimation_n):
    top_n_products = \
        avg_score_per_product[avg_score_per_product['product_id']\
                             .isin(category_products)]\
                             .sort_values(by='review_score', ascending=False)\
                             .iloc[:estimation_n]

    top_n_products = top_n_products.sample(n=product_number)
    
    return top_n_products

def get_expected_values(prediction_row, complete_data, products, n=5):
    # Calculates closest customers who bought the same product
    # Then gets estimated price, shipping cost and delivery time
    data = \
        get_customers_with_same_purchase(complete_data, products)
    customers_dict = get_closest_customers(prediction_row, data, products, n=n)

    expected_values = {}

    for product, customers in customers_dict.items():
        product_data = complete_data[complete_data['product_id']==product]
        price, shipping = estimate_price(customers, product_data)

        shipping_time = estimate_shipping(customers, product_data)

        expected_values_product = {'estimated_price': price,
                                   'shipping_price': shipping,
                                   'estimated_delivery_time': shipping_time}
        
        expected_values[product] = expected_values_product
    return expected_values

def get_closest_customers(prediction_row, data, products, n=5):
    # Get the 'n' closest customers who bought the same product
    customer_location = prediction_row[['customer_geolocation_lat',
                                        'customer_geolocation_lng']]
    
    customers_dict = {}

    for product in products:
        current_data = data[data['product_id']==product]

        location_data = current_data[['customer_unique_id',
                            'customer_geolocation_lat',
                            'customer_geolocation_lng']]
        
        location_data[['customer_geolocation_lat',
                       'customer_geolocation_lng']] = \
            (location_data[['customer_geolocation_lat',
                            'customer_geolocation_lng']] - customer_location).abs()

        location_data['absolute_diff'] = \
            location_data['customer_geolocation_lat'] + \
            location_data['customer_geolocation_lng']
        
        customers = location_data.sort_values(by='absolute_diff',
                                            ascending=True)\
                                 .iloc[:n]\
                                 .customer_unique_id.values
        customers_dict[product] = customers
    
    return customers_dict

def get_customers_with_same_purchase(complete_data, products):
    data = complete_data[complete_data['product_id']\
                .isin(products)]
    return data

def estimate_price(customers, complete_data):
    # Gets the mean price and freight for a subset of customers' purchases
    data = complete_data[complete_data['customer_unique_id'].isin(customers)]

    mean_price = data.price.mean()
    mean_freight = data.freight_value.mean()

    return mean_price, mean_freight

def estimate_shipping(customers, complete_data):
    data = complete_data[complete_data['customer_unique_id'].isin(customers)]
    purchase_time = 'order_purchase_timestamp'
    delivery_time = 'order_delivered_customer_date'
    date_format = '%Y-%m-%d %H:%M:%S'

    timedelta_avg = \
        (pd.to_datetime(data[delivery_time], format=date_format) - \
         pd.to_datetime(data[purchase_time], format=date_format)).mean()

    return timedelta_avg.days

