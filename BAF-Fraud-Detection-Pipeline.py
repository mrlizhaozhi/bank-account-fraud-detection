#!/usr/bin/env python
# coding: utf-8

# # Bank Account Fraud Detection Pipeline

# 

# In[46]:


# import required libraries
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


# In[47]:


# define required variable lists

# list of numeric features
numeric_features = [
    'name_email_similarity',
    'credit_risk_score',
    'proposed_credit_limit',
    'intended_balcon_amount',
    'prev_address_months_count',
    'date_of_birth_distinct_emails_4w',
    'current_address_months_count',
    'device_distinct_emails_8w',
    'income',
    'customer_age'
]

# list of catgorical features
categorical_features = [
    'prev_address_months_count_missing',
    'bank_months_count_missing',
    'intended_balcon_amount_missing',
    'foreign_request',
    'housing_status',
    'payment_type',
    'device_os',
    'keep_alive_session',
    'has_other_cards',
    'phone_home_valid',
    'source',
    'is_complete'
]

# list of columns with missing values.
missing_values_columns = [
    'prev_address_months_count',
    'current_address_months_count',
    'bank_months_count', 
    'device_distinct_emails_8w', 
    'intended_balcon_amount',
    'session_length_in_minutes'
]

missing_values_labels = [
    'prev_address_months_count_missing',
    'current_address_months_count_missing',
    'bank_months_count_missing', 
    'device_distinct_emails_8w_missing', 
    'intended_balcon_amount_missing',
    'session_length_in_minutes_missing'
]

# list of log transform candidate features
log_transform_candidates = [
    'proposed_credit_limit', 
    'intended_balcon_amount', 
    'current_address_months_count', 
    'prev_address_months_count', 
    'device_distinct_emails_8w'
]

# list of features to drop.
drop_features = [
    'intended_balcon_amount', 
    'prev_address_months_count', 
    'housing_status_BG', 
    'payment_type_AE',
    'payment_type_AC', 
    'intended_balcon_amount_missing_1', 
    'prev_address_months_count_missing_1', 
    'bank_months_count_missing_1'
]


# In[48]:


# impute missing values
class NegativesToNan(BaseEstimator, TransformerMixin):
    """ This class converts negatives into nan. """
    def __init__(self, columns):
        """ Initialise the class with columns with missing values. """
        self.columns = columns

    def fit(self, X, y=None):
        """Do nothing in fit."""
        self._fitted_ = True
        return self

    def transform(self, X):
        """Transform negatives into nan."""
        X = X.copy()
        for col in self.columns:
            if col in X.columns:
               X[col] = X[col].mask(X[col] < 0, np.nan)
        return X


# In[49]:


# create missing value labels
class MissingLabels(BaseEstimator, TransformerMixin):
    """This class creates missing value labels."""
    def __init__(self, columns):
        """Initialise the class with columns with missing values. """
        self.columns = columns

    def fit(self, X, y=None):
        """Do nothing in fit."""
        self._fitted_ = True
        return self

    def transform(self, X):
        """Transform missing values by creating new label columns."""
        X = X.copy()
        for col in self.columns:
            if col in X.columns:
                X[f'{col}_missing'] = (X[col] < 0).astype(int)
        return X


# In[50]:


# combine missing value indicators
class CombineMissingLabels(BaseEstimator, TransformerMixin):
    """This class combines missing value labels into one."""
    def __init__(self, columns):
        """Initialise the class with missing value label columns to be combined."""
        self.columns = columns

    def fit(self, X, y=None):
        """Do nothing in fit."""
        self._fitted_ = True
        return self

    def transform(self, X):
        """Combine missing labels into one."""
        X = X.copy()
        existing = [col for col in self.columns if col in X.columns]
        if existing:
            X['is_complete'] = X[existing].max(axis=1)
        return X


# In[51]:


# log transform all numeric features
class LogTransformer(BaseEstimator, TransformerMixin):
    """This class log transform numeric data."""
    def __init__(self, columns):
        """Initialise the class with columns to be log transformed."""
        self.columns = columns

    def fit(self, X, y=None):
        """Do nothing in fit."""
        self._fitted_ = True
        return self

    def transform(self, X):
        """Transform numeric data."""
        X = X.copy()
        for col in self.columns:
            if col in X.columns:
                X[col] = np.log1p(X[col])
        return X


# In[52]:


# drop unneeded features
class FeatureDropper(BaseEstimator, TransformerMixin):
    """This class drops unnecessary features."""
    def __init__(self, columns):
        """Initailise with features to drop."""
        self.columns = columns

    def fit(self, X, y=None):
        """Do nothing in fit."""
        self._fitted_ = True
        return self

    def transform(self, X):
        """Drop selected features."""
        X = X.copy()
        # drop only columns that exist
        existing = [col for col in self.columns if col in X.columns]

        return X.drop(columns=existing)


# In[53]:


# convert array to dataframe
class ArrayToDataFrame(BaseEstimator, TransformerMixin):
    """This class converts arrays to dataframes."""
    def __init__(self, columns):
        """Initialise with columns to transform."""
        self.columns = columns

    def fit(self, X, y=None):
        """Do nothing in fit."""
        self._fitted_ = True
        return self

    def transform(self, X):
        """Transform arrays into dataframes."""
        return pd.DataFrame(X, columns=self.columns)


# In[54]:


class ColumnTransformerToDataFrame(BaseEstimator, TransformerMixin):
    """
    Convert ColumnTransformer output to DataFrame safely.
    """

    def __init__(self, column_transformer):
        self.column_transformer = column_transformer

    def fit(self, X, y=None):
        self._fitted_= True
        feature_names = []

        # iterate through fitted transformers
        for name, transformer, columns in self.column_transformer.transformers_:

            if name == 'remainder':
                continue

            # numeric pipeline → keep original names
            if name == 'numeric_features_transformation':
                feature_names.extend(columns)

            # categorical pipeline → get names from fitted encoder
            elif name == 'categorical_features_transformation':

                encoder = transformer.named_steps['encoder']

                cat_names = encoder.get_feature_names_out(columns)

                feature_names.extend(cat_names)

        self.feature_names_ = feature_names

        return self

    def transform(self, X):

        return pd.DataFrame(X, columns=self.feature_names_)


# In[55]:


# numeric feature engineering pipeline
dataframe_pipeline = Pipeline([
    # create missing value labels 
    ('missing_values', MissingLabels(missing_values_columns)),
    # combine missing value indicator
    ('missing_indicator', CombineMissingLabels(missing_values_labels)),
    # convert negatives to nan
    ('negatives_to_nan', NegativesToNan(missing_values_columns)),
])


# In[56]:


# numeric feature engineering pipeline
numeric_features_pipeline = Pipeline([
    # impute nan with median
    ('simple_imputer', SimpleImputer(strategy='median')),
    # to dataframe
    ('to_dataframe', ArrayToDataFrame(numeric_features)),
    # log transform numeric features
    ('log_transformer', LogTransformer(log_transform_candidates)),
    # scale all data
    ('scale', StandardScaler())
])


# In[57]:


# categorical feature engineering pipeline
categorical_features_pipeline = Pipeline([

    ('encoder', OneHotEncoder(
        drop="first",
        handle_unknown="ignore",
        sparse_output=False
    ))

])


# In[58]:


# build preprocessor with ColumnTransformer
preprocessor = ColumnTransformer([
    # transform numeric features
    ('numeric_features_transformation', numeric_features_pipeline, numeric_features),
    # transform categorical features
    ('categorical_features_transformation', categorical_features_pipeline, categorical_features)
])


# In[59]:


# build the logistic regression model pipeline
logistic_regression_pipeline = Pipeline([
    # dataframe 
    ('dataframe', dataframe_pipeline),
    # preprocessor
    ('preprocessor', preprocessor),
    # to dataframe
    ('preprocessor_to_dataframe', ColumnTransformerToDataFrame(preprocessor)),
    # drop features
    ('drop_features', FeatureDropper(drop_features)),
])