from autogluon.tabular import TabularPredictor
import pandas as pd


# train_df = pd.read_csv('data/raw/train.csv')


def train_model(train_df):

    predictor = TabularPredictor(
        label='Survived',
        eval_metric='accuracy',
        problem_type='binary',  
    ).fit(
        train_data=train_df,  
        presets="optimize_for_deployment",  
        num_bag_folds=0,  
        num_stack_levels=0,  
        excluded_model_types=['KNN', 'RF', 'XT'],  
        time_limit=1200,  
        verbosity=2,
        auto_stack=True    
    )

    return predictor
