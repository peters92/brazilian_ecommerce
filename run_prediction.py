from pipeline import step3_prediction

predictor = step3_prediction.Predictor(config_path='config/step3_prediction.yaml')
#%%
# predictor.predict_category_random(sample_size=1)
#%%
# predictor.predict_product_random()
#%%
predictor.print_product_recommendation(sample_size=2, verbose=False,
                                       product_number=3)
