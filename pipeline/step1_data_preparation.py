import yaml
import pandas as pd
from sklearn.cluster import KMeans

from logic.data_preparation import (gather_customers_with_n_purchase,
                                    filter_complete_data_by_customer,
                                    build_predictor_table,
                                    prepare_modelling_data,
                                    prepare_unlabeled_data,
                                    prepare_product_recommendation_tables,
                                    join_coords)
from utils import save_data                                    


class DataPreparator():
    def __init__(self, data_path='data/raw/',
                 config_path='config/step1_data_preparation.yaml'): 
        
        # General attributes
        with open(config_path) as stream:
            self.config = yaml.safe_load(stream)
        self._load_data(data_path)
        self.complete_data_path = self.config['output']['complete_data_path']
        self.complete_data = None

        # Geolocation
        self._average_geolocation()
        self._kmeans_geolocation()

        # Classifier data attributes
        self.classifier_data_path = self.config['output']['classifier_data_path']
        self.columns_dict = self.config['control']['classifier_columns']
        self.classifier_columns = self._flatten_columns_dict()
        self.columns_path = self.config['output']['classifier_columns_path']
        self.classifier_data = None
        self.classifier_encoders = None
        self.encoder_path = self.config['output']['encoder_path']

        # Prodect recommendation
        self.product_recommendation_tables = None

        # Other data
        self.unlabeled_data_path = self.config['output']['unlabeled_data_path']
        

        # Debug
        self.predictor_table = None

    def _load_data(self, data_path):
        # Load all raw data files
        self.customers = pd.read_csv(data_path + self.config['input']['customers'])
        self.orders = pd.read_csv(data_path + self.config['input']['orders'])
        self.order_items = pd.read_csv(data_path + self.config['input']['order_items'])
        self.order_payments = pd.read_csv(data_path + self.config['input']['order_payments'])
        self.order_reviews = pd.read_csv(data_path + self.config['input']['order_reviews'])
        self.products = pd.read_csv(data_path + self.config['input']['products'])
        self.geolocation = pd.read_csv(data_path + self.config['input']['geolocation'])
        self.sellers = pd.read_csv(data_path + self.config['input']['sellers'])
        self.product_translation = \
            pd.read_csv(data_path + self.config['input']['product_translation'])

    def run(self):
        # Join data to create one table with all features for prediction
        self._join_data()
        # Translate the Portuguese product categories to English
        self._translate_product_categories()
        save_data(self.complete_data, self.complete_data_path)

        # Prepare data for modelling
        self._prepare_classifier_data()
        # Run unlabeled data through the same process and save for inference
        # later
        self._prepare_unlabeled_data()

        # Prepare processed tables for product recommendation
        self.product_recommendation_tables = \
            prepare_product_recommendation_tables(self.complete_data)

        save_data(self.classifier_data, self.classifier_data_path)
        save_data(self.unlabeled_data, self.unlabeled_data_path)
        save_data(self.classifier_encoders, self.encoder_path)
        save_data(self.columns_dict, self.columns_path)
        save_data(self.product_recommendation_tables,
                  self.config['output']['product_recommendation_table_path'])
    
    def _join_data(self):
        # Join all the relevant tables
        # Geolocation and product category translation is done separately
        tmp_pd = self.customers.merge(self.orders, on='customer_id', 
                                      how='outer')
        tmp_pd = tmp_pd.merge(self.order_items, how='outer')    
        tmp_pd = tmp_pd.merge(self.order_payments, how='outer')
        tmp_pd = tmp_pd.merge(self.order_reviews, how='outer')
        tmp_pd = tmp_pd.merge(self.products, on='product_id', how='outer')
        tmp_pd = tmp_pd.merge(self.sellers, on='seller_id', how='outer')


        tmp_pd = join_coords(tmp_pd, self.geolocation)

        tmp_pd = self._join_location(tmp_pd, join_column='location_cluster', 
                                     left_on='customer_zip_code_prefix',
                                     new_column_name='customer_location_cluster')
        tmp_pd = self._join_location(tmp_pd, join_column='location_cluster',
                                     left_on='seller_zip_code_prefix',
                                     new_column_name='seller_location_cluster')

        self.complete_data = tmp_pd

    def _translate_product_categories(self):
        tmp_pd = self.complete_data.merge(self.product_translation,
                                          on='product_category_name',
                                          how='outer')
        tmp_pd.drop(columns=['product_category_name'], inplace=True)
        tmp_pd.rename(columns={'product_category_name_english':
                               'product_category_name'},
                      inplace=True)

        self.complete_data = tmp_pd                                          

    def _prepare_classifier_data(self, n_purchase=2):
        # Get customers with multiple purchases
        # Their initial purchase will be used as a predictor for their 
        # follow-up purchase.
        relevant_customers = \
            gather_customers_with_n_purchase(data=self.complete_data,
                                             n=n_purchase)

        # Filter the complete data for these customers
        classifier_data = \
            filter_complete_data_by_customer(relevant_customers,
                                             data=self.complete_data,
                                             n=n_purchase)

        self.predictor_table = classifier_data
        # Organize data where each row contains columns as predictors
        # and the next purchase category as the target
        classifier_data = \
            build_predictor_table(classifier_data,
                                  columns=self.classifier_columns)
        classifier_data, standard_scaler, oh_encoder, label_encoder = \
            prepare_modelling_data(classifier_data, self.columns_dict)

        self.classifier_encoders = {'standard_scaler': standard_scaler,
                                    'one_hot_encoder': oh_encoder,
                                    'label_encoder': label_encoder}   
        self.classifier_data = classifier_data

    def _prepare_unlabeled_data(self):
        # Runs unlabeled data through the same processing as the labeled
        # (Except for targets)
        unlabeled_data = \
            prepare_unlabeled_data(self.complete_data,
                                   self.columns_dict,
                                   encoders=self.classifier_encoders)

        self.unlabeled_data = unlabeled_data

    def _average_geolocation(self):
        # Since zip codes have multiple entries for coordinates, and there is no
        # key showing which customer/seller it is, we have to average the
        # coordinates for each code.
        self.geolocation = self.geolocation.groupby('geolocation_zip_code_prefix')\
                                           .agg('mean')

    def _kmeans_geolocation(self):
        # Clusters the geolocation latitudes and longitudes
        # To reduce number of columns when location is later one hot encoded
        kmeans = \
            KMeans(n_clusters=self.config['control']['geolocation']['kmeans_clusters'])
        clusters = kmeans.fit_predict(\
            self.geolocation[['geolocation_lat', 'geolocation_lng']])

        self.geolocation['location_cluster'] = clusters
        self.geolocation.reset_index(inplace=True)

    def _join_location(self, data, join_column, left_on=None,
                       right_on='geolocation_zip_code_prefix',
                       new_column_name=None):
        # Joins the geolocation clusters to another table based on
        # zip code prefixes
        data = data.merge(self.geolocation[[right_on, join_column]], 
                          left_on=left_on,
                          right_on=right_on,
                          how='left')

        if new_column_name is not None:
            data.rename(columns={join_column: new_column_name},
                        inplace=True)
        
        return data

    def _flatten_columns_dict(self):
        flat_columns = []

        for columns in self.columns_dict.values():
            for column in columns:
                flat_columns.append(column)
        
        return flat_columns
