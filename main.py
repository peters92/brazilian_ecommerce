#%%
# %reload_ext autoreload
# %autoreload 2
from pipeline import (step1_data_preparation,
                      step2_modelling,
                      step3_prediction)
#%%
data_preparator = \
    step1_data_preparation.DataPreparator(config_path='config/step1_data_preparation.yaml')
#%%
data_preparator.run()
#%%
modeller = step2_modelling.Modeller(config_path='config/step2_modelling.yaml')
modeller.run()
#%%
predictor = step3_prediction.Predictor(config_path='config/step3_prediction.yaml')
#%%
# predictor.predict_category_random(sample_size=1)
#%%
# predictor.predict_product_random()
#%%
predictor.print_product_recommendation(sample_size=5, verbose=False,
                                       product_number=5)

