import yaml
from utils import load_data, trim_id
from logic.data_preparation import prepare_unlabeled_data
from logic.prediction import get_recommended_product 


class Predictor():
    def __init__(self, config_path='config/step3_prediction.yaml'):
        # General attributes
        with open(config_path) as stream:
            self.config = yaml.safe_load(stream)
        
        self.complete_data = \
            load_data(self.config['input']['complete_data_path'])
        self.data = load_data(self.config['input']['data_path'])
        self.model = load_data(self.config['input']['model_path'])
        self.encoders = load_data(self.config['input']['encoder_path'])
        self.columns_dict = \
            load_data(self.config['input']['classifier_columns_path'])

        self.product_recommendation_tables = \
            load_data(self.config['input']['product_recommendation_table_path'])

    def predict_category_random(self, sample_size=5, to_print=True):
        sample = self.complete_data.sample(n=sample_size, random_state=None)
        # Generates product category prediction for a random sample 
        # of the complete data
        prepared_data = prepare_unlabeled_data(sample, self.columns_dict, 
                                               self.encoders)
        
        if to_print:
            print(f'Running prediction for sample:\n{sample}')
        predictions = self.model.predict(prepared_data)
        
        label_encoder = self.encoders['label_encoder']
        predictions_string = \
            label_encoder.inverse_transform(predictions.astype(int))
        if to_print:
            print(f'\nPredictions:\n{predictions_string}')
        
        sample['predicted_category'] = predictions_string

        return sample

    def predict_product_random(self, sample_size=2, product_number=3, estimation_n=5):
        category_predictions = \
            self.predict_category_random(sample_size=sample_size,
                                         to_print=False)
        recommendations = {}
        for _, row in category_predictions.iterrows():
            recommendation = get_recommended_product(row, self.complete_data,
                                             self.product_recommendation_tables,
                                             product_number=product_number,
                                             estimation_n=estimation_n)

            recommendations[row['customer_unique_id']] = recommendation

        return category_predictions, recommendations

    def print_product_recommendation(self, sample_size=2, verbose=False,
                                     product_number=3, estimation_number=5):
        # Prints predictions per customer for a given sample size
        # Based on verbosity it will either print previous category, predicted
        # category and recommended product information, or it will print all the
        # customer's order info and other features as well.
        category_prediction, recommendations = \
            self.predict_product_random(sample_size=sample_size,
                                        product_number=product_number,
                                        estimation_n=1000)

        # Loop through customers
        for _, row in category_prediction.iterrows():
            customer = row['customer_unique_id']
            previous_category = row['product_category_name']
            predicted_category = row['predicted_category']
            rec_customer = recommendations[customer]

            left_column = 35
            right_column = 15
            print(f'{"Predictions for customer:":<{left_column}}'
                  f'{trim_id(customer):>{right_column}}')
            print(f'{"Last purchase product category:":<{left_column}}'
                  f'{previous_category:>{right_column}}')
            print(f'{"Predicted next product category:":<{left_column}}'
                  f'{predicted_category:>{right_column}}')

            if verbose:
                print(f'\nComplete data for order:\n{row}\n')

            print(f'\n{"  Recommended products  ":#^50}')
            review_scores = rec_customer['product_scores']
            estimations = rec_customer['expected_values']
            for product, estimation in estimations.items():
                price = estimation['estimated_price']
                shipping = estimation['shipping_price']
                shipping_time = estimation['estimated_delivery_time']
                review_score = review_scores[product]
                print(f'{"Product:":<{left_column}}'f'{trim_id(product):>{right_column}}')
                print(f'{"Price + estimated shipping cost:":<{left_column}}'
                      f'{f"{price:.2f} + {shipping:.2f}R$":>{right_column}}')
                print(f'{"Estimated delivery time:":<{left_column}}'
                      f'{f"{shipping_time} days":>{right_column}}')
                print(f'{"Avg. review score:":<{left_column}}'
                      f'{review_score:>{right_column}.2f}')
            
            print(f'{"":#^50}\n\n')

