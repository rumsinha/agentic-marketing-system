#!/usr/bin/env python3
"""
FEATURE ENGINEERING FOR CUSTOMER CLUSTERING
============================================
This script creates features optimized for customer segmentation using
both K-Means and HDBSCAN clustering algorithms.

Feature Categories:
1. RFM Features (Recency, Frequency, Monetary)
2. Behavioral Features (categories, products, payment patterns)
3. Engagement Features (reviews, ratings)
4. Temporal Features (purchase timing patterns)
5. Geographic Features (location-based)

Usage:
    python feature_engineering.py --db-url postgresql://... --output-dir ./features

Author: Marketing AI Platform
"""

import os
import sys
import argparse
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sqlalchemy import create_engine, text

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature engineering pipeline for customer clustering.
    Creates a comprehensive feature set from raw Olist data.
    """
    
    def __init__(self, db_url: str, output_dir: str = './features'):
        """
        Initialize feature engineer.
        
        Args:
            db_url: PostgreSQL connection string
            output_dir: Directory to save features and transformers
        """
        self.db_url = db_url
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.engine = None
        self.reference_date = None
        self.transformers = {}
        
    def connect(self) -> bool:
        """Establish database connection."""
        try:
            self.engine = create_engine(self.db_url)
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Connected to database")
            return True
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False
    
    # =========================================================================
    # DATA LOADING
    # =========================================================================
    
    def load_base_data(self) -> pd.DataFrame:
        """
        Load and join all necessary tables for feature engineering.
        Returns a denormalized dataset at customer level.
        """
        logger.info("Loading base data...")
        
        query = """
        WITH customer_orders AS (
            SELECT 
                c.customer_unique_id,
                c.customer_city,
                c.customer_state,
                c.customer_zip_code_prefix,
                o.order_id,
                o.order_status,
                o.order_purchase_timestamp,
                o.order_approved_at,
                o.order_delivered_carrier_date,
                o.order_delivered_customer_date,
                o.order_estimated_delivery_date
            FROM olist.customers c
            JOIN olist.orders o ON c.customer_id = o.customer_id
        ),
        order_details AS (
            SELECT 
                co.*,
                oi.product_id,
                oi.seller_id,
                oi.price,
                oi.freight_value,
                p.product_category_name,
                pct.product_category_name_english as category_english,
                p.product_weight_g,
                p.product_photos_qty
            FROM customer_orders co
            JOIN olist.order_items oi ON co.order_id = oi.order_id
            LEFT JOIN olist.products p ON oi.product_id = p.product_id
            LEFT JOIN olist.product_category_translation pct 
                ON p.product_category_name = pct.product_category_name
        ),
        order_payments AS (
            SELECT 
                order_id,
                payment_type,
                payment_installments,
                payment_value
            FROM olist.order_payments
        ),
        order_reviews AS (
            SELECT 
                order_id,
                review_score,
                review_comment_message,
                CASE WHEN review_comment_message IS NOT NULL THEN 1 ELSE 0 END as has_review_text,
                COALESCE(LENGTH(review_comment_message), 0) as review_text_length
            FROM olist.order_reviews
        )
        SELECT 
            od.*,
            op.payment_type,
            op.payment_installments,
            op.payment_value,
            orv.review_score,
            orv.has_review_text,
            orv.review_text_length
        FROM order_details od
        LEFT JOIN order_payments op ON od.order_id = op.order_id
        LEFT JOIN order_reviews orv ON od.order_id = orv.order_id
        """
        
        df = pd.read_sql(query, self.engine)
        logger.info(f"Loaded {len(df):,} rows")
        
        # Convert timestamps
        timestamp_cols = ['order_purchase_timestamp', 'order_approved_at', 
                         'order_delivered_carrier_date', 'order_delivered_customer_date',
                         'order_estimated_delivery_date']
        for col in timestamp_cols:
            df[col] = pd.to_datetime(df[col])
        
        # Set reference date (max date + 1 day)
        self.reference_date = df['order_purchase_timestamp'].max() + pd.Timedelta(days=1)
        logger.info(f"Reference date: {self.reference_date}")
        
        return df
    
    # =========================================================================
    # FEATURE ENGINEERING - RFM
    # =========================================================================
    
    def create_rfm_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create RFM (Recency, Frequency, Monetary) features.
        These are the foundation of customer segmentation.
        """
        logger.info("Creating RFM features...")
        
        # Aggregate to customer level
        rfm = df.groupby('customer_unique_id').agg({
            # Recency: days since last purchase
            'order_purchase_timestamp': lambda x: (self.reference_date - x.max()).days,
            # Frequency: number of unique orders
            'order_id': 'nunique',
            # Monetary: total spend
            'price': 'sum',
            'freight_value': 'sum'
        }).reset_index()
        
        rfm.columns = ['customer_unique_id', 'recency', 'frequency', 'total_price', 'total_freight']
        
        # Calculate monetary (total spend)
        rfm['monetary'] = rfm['total_price'] + rfm['total_freight']
        
        # Average order value
        rfm['avg_order_value'] = rfm['monetary'] / rfm['frequency']
        
        # Average items per order
        items_per_order = df.groupby(['customer_unique_id', 'order_id']).size().reset_index(name='items')
        avg_items = items_per_order.groupby('customer_unique_id')['items'].mean().reset_index()
        avg_items.columns = ['customer_unique_id', 'avg_items_per_order']
        rfm = rfm.merge(avg_items, on='customer_unique_id', how='left')
        
        # Tenure: days since first purchase
        first_purchase = df.groupby('customer_unique_id')['order_purchase_timestamp'].min().reset_index()
        first_purchase.columns = ['customer_unique_id', 'first_purchase_date']
        first_purchase['tenure_days'] = (self.reference_date - first_purchase['first_purchase_date']).dt.days
        rfm = rfm.merge(first_purchase[['customer_unique_id', 'tenure_days']], 
                       on='customer_unique_id', how='left')
        
        # Purchase frequency (orders per month of tenure)
        rfm['purchase_frequency_monthly'] = rfm['frequency'] / (rfm['tenure_days'] / 30).clip(lower=1)
        
        # Inter-purchase time (average days between orders, for repeat customers)
        def calc_inter_purchase_time(group):
            if len(group) <= 1:
                return np.nan
            dates = group.sort_values()
            return dates.diff().dt.days.mean()
        
        inter_purchase = df.groupby('customer_unique_id')['order_purchase_timestamp'].apply(
            calc_inter_purchase_time
        ).reset_index()
        inter_purchase.columns = ['customer_unique_id', 'avg_inter_purchase_days']
        rfm = rfm.merge(inter_purchase, on='customer_unique_id', how='left')
        
        # Drop intermediate columns
        rfm = rfm.drop(columns=['total_price', 'total_freight'])
        
        logger.info(f"  Created {len(rfm.columns) - 1} RFM features")
        return rfm
    
    # =========================================================================
    # FEATURE ENGINEERING - BEHAVIORAL
    # =========================================================================
    
    def create_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create behavioral features based on purchase patterns.
        """
        logger.info("Creating behavioral features...")
        
        # --- Category Features ---
        
        # Number of distinct categories purchased
        category_count = df.groupby('customer_unique_id')['category_english'].nunique().reset_index()
        category_count.columns = ['customer_unique_id', 'distinct_categories']
        
        # Favorite category (mode)
        def get_favorite_category(group):
            mode = group['category_english'].mode()
            return mode.iloc[0] if len(mode) > 0 else 'unknown'
        
        favorite_cat = df.groupby('customer_unique_id').apply(
            get_favorite_category, include_groups=False
        ).reset_index()
        favorite_cat.columns = ['customer_unique_id', 'favorite_category']
        
        # Category concentration (how focused are purchases)
        def category_concentration(group):
            counts = group['category_english'].value_counts(normalize=True)
            return (counts ** 2).sum()  # Herfindahl index
        
        cat_concentration = df.groupby('customer_unique_id').apply(
            category_concentration, include_groups=False
        ).reset_index()
        cat_concentration.columns = ['customer_unique_id', 'category_concentration']
        
        # --- Product Features ---
        
        # Number of distinct products
        product_count = df.groupby('customer_unique_id')['product_id'].nunique().reset_index()
        product_count.columns = ['customer_unique_id', 'distinct_products']
        
        # Average product weight
        avg_weight = df.groupby('customer_unique_id')['product_weight_g'].mean().reset_index()
        avg_weight.columns = ['customer_unique_id', 'avg_product_weight']
        
        # Average product photos (proxy for product quality/detail)
        avg_photos = df.groupby('customer_unique_id')['product_photos_qty'].mean().reset_index()
        avg_photos.columns = ['customer_unique_id', 'avg_product_photos']
        
        # --- Price Features ---
        
        # Average item price
        avg_price = df.groupby('customer_unique_id')['price'].mean().reset_index()
        avg_price.columns = ['customer_unique_id', 'avg_item_price']
        
        # Price variance (do they buy across price ranges?)
        price_std = df.groupby('customer_unique_id')['price'].std().reset_index()
        price_std.columns = ['customer_unique_id', 'price_std']
        
        # Max single item price
        max_price = df.groupby('customer_unique_id')['price'].max().reset_index()
        max_price.columns = ['customer_unique_id', 'max_item_price']
        
        # Freight ratio (freight / total value)
        freight_stats = df.groupby('customer_unique_id').agg({
            'freight_value': 'sum',
            'price': 'sum'
        }).reset_index()
        freight_stats['freight_ratio'] = freight_stats['freight_value'] / (
            freight_stats['freight_value'] + freight_stats['price']
        ).replace(0, np.nan)
        freight_stats = freight_stats[['customer_unique_id', 'freight_ratio']]
        
        # --- Seller Features ---
        
        # Number of distinct sellers
        seller_count = df.groupby('customer_unique_id')['seller_id'].nunique().reset_index()
        seller_count.columns = ['customer_unique_id', 'distinct_sellers']
        
        # Merge all behavioral features
        behavioral = category_count
        for feature_df in [favorite_cat, cat_concentration, product_count, 
                          avg_weight, avg_photos, avg_price, price_std, 
                          max_price, freight_stats, seller_count]:
            behavioral = behavioral.merge(feature_df, on='customer_unique_id', how='left')
        
        logger.info(f"  Created {len(behavioral.columns) - 1} behavioral features")
        return behavioral
    
    # =========================================================================
    # FEATURE ENGINEERING - PAYMENT
    # =========================================================================
    
    def create_payment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features based on payment behavior.
        """
        logger.info("Creating payment features...")
        
        # Primary payment method
        def get_primary_payment(group):
            mode = group['payment_type'].mode()
            return mode.iloc[0] if len(mode) > 0 else 'unknown'
        
        primary_payment = df.groupby('customer_unique_id').apply(
            get_primary_payment, include_groups=False
        ).reset_index()
        primary_payment.columns = ['customer_unique_id', 'primary_payment_method']
        
        # Payment method diversity
        payment_diversity = df.groupby('customer_unique_id')['payment_type'].nunique().reset_index()
        payment_diversity.columns = ['customer_unique_id', 'payment_method_diversity']
        
        # Average installments (for credit card users)
        credit_card_df = df[df['payment_type'] == 'credit_card']
        avg_installments = credit_card_df.groupby('customer_unique_id')['payment_installments'].mean().reset_index()
        avg_installments.columns = ['customer_unique_id', 'avg_installments']
        
        # Max installments used
        max_installments = credit_card_df.groupby('customer_unique_id')['payment_installments'].max().reset_index()
        max_installments.columns = ['customer_unique_id', 'max_installments']
        
        # Credit card usage rate
        payment_counts = df.groupby('customer_unique_id')['payment_type'].value_counts().unstack(fill_value=0)
        payment_counts['total_payments'] = payment_counts.sum(axis=1)
        if 'credit_card' in payment_counts.columns:
            payment_counts['credit_card_rate'] = payment_counts['credit_card'] / payment_counts['total_payments']
        else:
            payment_counts['credit_card_rate'] = 0
        credit_card_rate = payment_counts[['credit_card_rate']].reset_index()
        
        # Boleto usage rate
        if 'boleto' in payment_counts.columns:
            payment_counts['boleto_rate'] = payment_counts['boleto'] / payment_counts['total_payments']
        else:
            payment_counts['boleto_rate'] = 0
        boleto_rate = payment_counts[['boleto_rate']].reset_index()
        
        # Merge payment features
        payment = primary_payment
        for feature_df in [payment_diversity, avg_installments, max_installments, 
                          credit_card_rate, boleto_rate]:
            payment = payment.merge(feature_df, on='customer_unique_id', how='left')
        
        # Fill NaN for non-credit card users
        payment['avg_installments'] = payment['avg_installments'].fillna(0)
        payment['max_installments'] = payment['max_installments'].fillna(0)
        
        logger.info(f"  Created {len(payment.columns) - 1} payment features")
        return payment
    
    # =========================================================================
    # FEATURE ENGINEERING - ENGAGEMENT (Reviews)
    # =========================================================================
    
    def create_engagement_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features based on review and engagement behavior.
        """
        logger.info("Creating engagement features...")
        
        # Review rate (% of orders with reviews)
        review_data = df.groupby('customer_unique_id').agg({
            'review_score': ['count', 'mean', 'std'],
            'has_review_text': 'sum',
            'review_text_length': 'mean',
            'order_id': 'nunique'
        })
        review_data.columns = ['review_count', 'avg_review_score', 'review_score_std',
                               'reviews_with_text', 'avg_review_length', 'total_orders']
        review_data = review_data.reset_index()
        
        # Review engagement rate
        review_data['review_rate'] = review_data['review_count'] / review_data['total_orders']
        
        # Text review rate (among reviews)
        review_data['text_review_rate'] = review_data['reviews_with_text'] / review_data['review_count'].replace(0, np.nan)
        
        # Sentiment proxy (based on scores)
        # Low scores (1-2) = negative, High scores (4-5) = positive
        def sentiment_ratio(group):
            scores = group['review_score'].dropna()
            if len(scores) == 0:
                return np.nan
            positive = (scores >= 4).sum()
            negative = (scores <= 2).sum()
            total = len(scores)
            return (positive - negative) / total
        
        sentiment = df.groupby('customer_unique_id').apply(
            sentiment_ratio, include_groups=False
        ).reset_index()
        sentiment.columns = ['customer_unique_id', 'sentiment_ratio']
        
        # Reviewer type (based on review patterns)
        def reviewer_type(row):
            if pd.isna(row['avg_review_score']):
                return 'non_reviewer'
            elif row['text_review_rate'] > 0.5:
                return 'detailed_reviewer'
            elif row['avg_review_score'] >= 4:
                return 'positive_reviewer'
            elif row['avg_review_score'] <= 2:
                return 'critical_reviewer'
            else:
                return 'neutral_reviewer'
        
        engagement = review_data[['customer_unique_id', 'avg_review_score', 'review_score_std',
                                  'review_rate', 'text_review_rate', 'avg_review_length']]
        engagement = engagement.merge(sentiment, on='customer_unique_id', how='left')
        
        # Fill NaN values
        engagement['avg_review_score'] = engagement['avg_review_score'].fillna(0)
        engagement['review_score_std'] = engagement['review_score_std'].fillna(0)
        engagement['text_review_rate'] = engagement['text_review_rate'].fillna(0)
        engagement['avg_review_length'] = engagement['avg_review_length'].fillna(0)
        engagement['sentiment_ratio'] = engagement['sentiment_ratio'].fillna(0)
        
        logger.info(f"  Created {len(engagement.columns) - 1} engagement features")
        return engagement
    
    # =========================================================================
    # FEATURE ENGINEERING - TEMPORAL
    # =========================================================================
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features based on purchase timing patterns.
        """
        logger.info("Creating temporal features...")
        
        df = df.copy()
        df['order_dow'] = df['order_purchase_timestamp'].dt.dayofweek
        df['order_hour'] = df['order_purchase_timestamp'].dt.hour
        df['order_month'] = df['order_purchase_timestamp'].dt.month
        df['is_weekend'] = df['order_dow'].isin([5, 6]).astype(int)
        
        # Preferred day of week
        preferred_dow = df.groupby('customer_unique_id')['order_dow'].agg(
            lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else -1
        ).reset_index()
        preferred_dow.columns = ['customer_unique_id', 'preferred_dow']
        
        # Preferred hour
        preferred_hour = df.groupby('customer_unique_id')['order_hour'].agg(
            lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else -1
        ).reset_index()
        preferred_hour.columns = ['customer_unique_id', 'preferred_hour']
        
        # Weekend shopper rate
        weekend_rate = df.groupby('customer_unique_id')['is_weekend'].mean().reset_index()
        weekend_rate.columns = ['customer_unique_id', 'weekend_shopping_rate']
        
        # Time of day preference (morning/afternoon/evening/night)
        def time_of_day(hour):
            if 6 <= hour < 12:
                return 'morning'
            elif 12 <= hour < 18:
                return 'afternoon'
            elif 18 <= hour < 22:
                return 'evening'
            else:
                return 'night'
        
        df['time_of_day'] = df['order_hour'].apply(time_of_day)
        preferred_tod = df.groupby('customer_unique_id')['time_of_day'].agg(
            lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'unknown'
        ).reset_index()
        preferred_tod.columns = ['customer_unique_id', 'preferred_time_of_day']
        
        # Delivery experience features
        delivered_df = df[df['order_status'] == 'delivered'].copy()
        delivered_df['delivery_days'] = (
            delivered_df['order_delivered_customer_date'] - 
            delivered_df['order_purchase_timestamp']
        ).dt.days
        delivered_df['delivery_delay'] = (
            delivered_df['order_delivered_customer_date'] - 
            delivered_df['order_estimated_delivery_date']
        ).dt.days
        
        delivery_stats = delivered_df.groupby('customer_unique_id').agg({
            'delivery_days': 'mean',
            'delivery_delay': 'mean'
        }).reset_index()
        delivery_stats.columns = ['customer_unique_id', 'avg_delivery_days', 'avg_delivery_delay']
        
        # Merge temporal features
        temporal = preferred_dow
        for feature_df in [preferred_hour, weekend_rate, preferred_tod, delivery_stats]:
            temporal = temporal.merge(feature_df, on='customer_unique_id', how='left')
        
        # Fill NaN
        temporal['avg_delivery_days'] = temporal['avg_delivery_days'].fillna(temporal['avg_delivery_days'].median())
        temporal['avg_delivery_delay'] = temporal['avg_delivery_delay'].fillna(0)
        
        logger.info(f"  Created {len(temporal.columns) - 1} temporal features")
        return temporal
    
    # =========================================================================
    # FEATURE ENGINEERING - GEOGRAPHIC
    # =========================================================================
    
    def create_geographic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features based on customer location.
        """
        logger.info("Creating geographic features...")
        
        # Get unique customer location
        customer_location = df.groupby('customer_unique_id').agg({
            'customer_state': 'first',
            'customer_city': 'first',
            'customer_zip_code_prefix': 'first'
        }).reset_index()
        
        # State-level features
        # Calculate state statistics
        state_stats = df.groupby('customer_state').agg({
            'customer_unique_id': 'nunique',
            'price': 'mean'
        }).reset_index()
        state_stats.columns = ['customer_state', 'state_customer_count', 'state_avg_price']
        
        # Region mapping (Brazilian regions)
        region_map = {
            'AC': 'Norte', 'AP': 'Norte', 'AM': 'Norte', 'PA': 'Norte', 
            'RO': 'Norte', 'RR': 'Norte', 'TO': 'Norte',
            'AL': 'Nordeste', 'BA': 'Nordeste', 'CE': 'Nordeste', 'MA': 'Nordeste',
            'PB': 'Nordeste', 'PE': 'Nordeste', 'PI': 'Nordeste', 'RN': 'Nordeste', 'SE': 'Nordeste',
            'DF': 'Centro-Oeste', 'GO': 'Centro-Oeste', 'MT': 'Centro-Oeste', 'MS': 'Centro-Oeste',
            'ES': 'Sudeste', 'MG': 'Sudeste', 'RJ': 'Sudeste', 'SP': 'Sudeste',
            'PR': 'Sul', 'RS': 'Sul', 'SC': 'Sul'
        }
        customer_location['region'] = customer_location['customer_state'].map(region_map)
        
        # Is major city (São Paulo, Rio de Janeiro)
        major_cities = ['sao paulo', 'rio de janeiro', 'belo horizonte', 'brasilia', 'curitiba']
        customer_location['is_major_city'] = customer_location['customer_city'].str.lower().isin(major_cities).astype(int)
        
        # Merge state stats
        geographic = customer_location.merge(state_stats, on='customer_state', how='left')
        
        # Rename and select final columns
        geographic = geographic.rename(columns={'customer_state': 'state', 'customer_city': 'city'})
        geographic = geographic[['customer_unique_id', 'state', 'region', 'is_major_city',
                                 'state_customer_count', 'state_avg_price']]
        
        logger.info(f"  Created {len(geographic.columns) - 1} geographic features")
        return geographic
    
    # =========================================================================
    # FEATURE COMBINATION AND TRANSFORMATION
    # =========================================================================
    
    def combine_features(
        self,
        rfm: pd.DataFrame,
        behavioral: pd.DataFrame,
        payment: pd.DataFrame,
        engagement: pd.DataFrame,
        temporal: pd.DataFrame,
        geographic: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Combine all feature sets into a single dataframe.
        """
        logger.info("Combining all features...")
        
        # Start with RFM (has all customers)
        features = rfm
        
        # Merge other feature sets
        for feature_df in [behavioral, payment, engagement, temporal, geographic]:
            features = features.merge(feature_df, on='customer_unique_id', how='left')
        
        logger.info(f"  Combined dataset: {features.shape[0]:,} customers, {features.shape[1]} features")
        return features
    
    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features for clustering.
        """
        logger.info("Encoding categorical features...")
        
        df = df.copy()
        categorical_cols = ['favorite_category', 'primary_payment_method', 
                          'preferred_time_of_day', 'state', 'region']
        
        for col in categorical_cols:
            if col in df.columns:
                # Create frequency encoding (good for clustering)
                freq_map = df[col].value_counts(normalize=True).to_dict()
                df[f'{col}_freq'] = df[col].map(freq_map)
                
                # Create label encoding for models that need it
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col].fillna('unknown'))
                
                # Store encoder
                self.transformers[f'{col}_encoder'] = le
        
        logger.info(f"  Encoded {len(categorical_cols)} categorical features")
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the feature set.
        """
        logger.info("Handling missing values...")
        
        df = df.copy()
        
        # Report missing values
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if len(missing) > 0:
            logger.info(f"  Missing values before handling:\n{missing}")
        
        # Fill strategies based on feature type
        fill_strategies = {
            # Fill with median for numeric
            'avg_inter_purchase_days': df['avg_inter_purchase_days'].median(),
            'avg_review_score': 0,
            'review_score_std': 0,
            'avg_review_length': 0,
            'sentiment_ratio': 0,
            'avg_delivery_days': df['avg_delivery_days'].median(),
            'avg_delivery_delay': 0,
            'category_concentration': 1,  # Single category = 1
            'price_std': 0,
            'avg_product_weight': df['avg_product_weight'].median(),
            'avg_product_photos': df['avg_product_photos'].median(),
        }
        
        for col, fill_value in fill_strategies.items():
            if col in df.columns and df[col].isnull().any():
                df[col] = df[col].fillna(fill_value)
        
        # Fill remaining numeric with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
        
        # Fill remaining categorical with 'unknown'
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            df[col] = df[col].fillna('unknown')
        
        logger.info(f"  Missing values after handling: {df.isnull().sum().sum()}")
        return df
    
    def scale_features(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        method: str = 'robust'
    ) -> Tuple[pd.DataFrame, object]:
        """
        Scale numeric features for clustering.
        
        Args:
            df: DataFrame with features
            feature_cols: List of columns to scale
            method: 'standard', 'robust', or 'power'
            
        Returns:
            Scaled DataFrame and the scaler object
        """
        logger.info(f"Scaling features using {method} method...")
        
        df = df.copy()
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        elif method == 'power':
            scaler = PowerTransformer(method='yeo-johnson')
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        # Fit and transform
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
        
        self.transformers['scaler'] = scaler
        self.transformers['scaled_columns'] = feature_cols
        
        logger.info(f"  Scaled {len(feature_cols)} features")
        return df, scaler
    
    def select_features_for_clustering(self, df: pd.DataFrame) -> List[str]:
        """
        Select the best features for clustering.
        Returns a list of feature column names.
        """
        logger.info("Selecting features for clustering...")
        
        # Core RFM features (always include)
        rfm_features = [
            'recency', 'frequency', 'monetary', 'avg_order_value',
            'avg_items_per_order', 'tenure_days', 'purchase_frequency_monthly'
        ]
        
        # Behavioral features
        behavioral_features = [
            'distinct_categories', 'category_concentration', 'distinct_products',
            'avg_item_price', 'price_std', 'max_item_price', 'freight_ratio',
            'distinct_sellers'
        ]
        
        # Payment features
        payment_features = [
            'payment_method_diversity', 'avg_installments', 'credit_card_rate'
        ]
        
        # Engagement features
        engagement_features = [
            'avg_review_score', 'review_rate', 'text_review_rate', 'sentiment_ratio'
        ]
        
        # Temporal features
        temporal_features = [
            'weekend_shopping_rate', 'avg_delivery_days'
        ]
        
        # Geographic features (encoded)
        geographic_features = [
            'is_major_city', 'region_freq'
        ]
        
        # Combine all
        all_features = (rfm_features + behavioral_features + payment_features + 
                       engagement_features + temporal_features + geographic_features)
        
        # Filter to only existing columns
        selected_features = [f for f in all_features if f in df.columns]
        
        logger.info(f"  Selected {len(selected_features)} features for clustering")
        return selected_features
    
    # =========================================================================
    # MAIN PIPELINE
    # =========================================================================
    
    def run_pipeline(self) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        """
        Run the complete feature engineering pipeline.
        
        Returns:
            - Full feature DataFrame (unscaled)
            - Scaled feature DataFrame
            - List of feature columns used for clustering
        """
        logger.info("=" * 60)
        logger.info("STARTING FEATURE ENGINEERING PIPELINE")
        logger.info("=" * 60)
        
        # Load data
        df = self.load_base_data()
        
        # Create feature sets
        rfm = self.create_rfm_features(df)
        behavioral = self.create_behavioral_features(df)
        payment = self.create_payment_features(df)
        engagement = self.create_engagement_features(df)
        temporal = self.create_temporal_features(df)
        geographic = self.create_geographic_features(df)
        
        # Combine features
        features = self.combine_features(rfm, behavioral, payment, engagement, 
                                        temporal, geographic)
        
        # Encode categorical features
        features = self.encode_categorical_features(features)
        
        # Handle missing values
        features = self.handle_missing_values(features)
        
        # Select features for clustering
        feature_cols = self.select_features_for_clustering(features)
        
        # Create scaled version
        features_scaled = features.copy()
        features_scaled, scaler = self.scale_features(
            features_scaled, 
            feature_cols, 
            method='robust'
        )
        
        # Save outputs
        logger.info("Saving outputs...")
        
        # Save feature data
        features.to_csv(self.output_dir / 'customer_features.csv', index=False)
        features_scaled.to_csv(self.output_dir / 'customer_features_scaled.csv', index=False)
        
        # Save transformers
        with open(self.output_dir / 'transformers.pkl', 'wb') as f:
            pickle.dump(self.transformers, f)
        
        # Save feature list
        with open(self.output_dir / 'feature_columns.txt', 'w') as f:
            f.write('\n'.join(feature_cols))
        
        # Generate feature summary
        self.generate_feature_summary(features, feature_cols)
        
        logger.info("=" * 60)
        logger.info("FEATURE ENGINEERING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"  Total customers: {len(features):,}")
        logger.info(f"  Total features: {len(features.columns)}")
        logger.info(f"  Clustering features: {len(feature_cols)}")
        logger.info(f"  Outputs saved to: {self.output_dir}")
        
        return features, features_scaled, feature_cols
    
    def generate_feature_summary(self, df: pd.DataFrame, feature_cols: List[str]):
        """Generate a summary of features for documentation."""
        summary = df[feature_cols].describe().T
        summary['missing'] = df[feature_cols].isnull().sum()
        summary['dtype'] = df[feature_cols].dtypes
        
        summary.to_csv(self.output_dir / 'feature_summary.csv')
        
        # Generate markdown documentation
        doc = f"""# Feature Engineering Summary
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Dataset Statistics
- **Customers**: {len(df):,}
- **Total Features**: {len(df.columns)}
- **Clustering Features**: {len(feature_cols)}

## Feature List for Clustering

### RFM Features
| Feature | Description |
|---------|-------------|
| recency | Days since last purchase |
| frequency | Number of orders |
| monetary | Total spend (BRL) |
| avg_order_value | Average order value |
| avg_items_per_order | Average items per order |
| tenure_days | Days since first purchase |
| purchase_frequency_monthly | Orders per month |

### Behavioral Features
| Feature | Description |
|---------|-------------|
| distinct_categories | Number of unique categories purchased |
| category_concentration | Herfindahl index of category purchases |
| distinct_products | Number of unique products purchased |
| avg_item_price | Average price of items purchased |
| price_std | Standard deviation of item prices |
| max_item_price | Maximum item price purchased |
| freight_ratio | Freight as % of total spend |
| distinct_sellers | Number of unique sellers purchased from |

### Payment Features
| Feature | Description |
|---------|-------------|
| payment_method_diversity | Number of different payment methods used |
| avg_installments | Average installments (credit card) |
| credit_card_rate | % of payments via credit card |

### Engagement Features
| Feature | Description |
|---------|-------------|
| avg_review_score | Average review rating given |
| review_rate | % of orders with reviews |
| text_review_rate | % of reviews with text comments |
| sentiment_ratio | Positive - Negative sentiment ratio |

### Temporal Features
| Feature | Description |
|---------|-------------|
| weekend_shopping_rate | % of purchases on weekends |
| avg_delivery_days | Average delivery time experienced |

### Geographic Features
| Feature | Description |
|---------|-------------|
| is_major_city | 1 if in major city, 0 otherwise |
| region_freq | Frequency encoding of region |

## Feature Statistics
{summary.to_markdown()}
"""
        
        with open(self.output_dir / 'feature_documentation.md', 'w') as f:
            f.write(doc)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Olist Feature Engineering')
    parser.add_argument('--db-url', type=str, required=True,
                       help='PostgreSQL connection URL')
    parser.add_argument('--output-dir', type=str, default='./features',
                       help='Output directory for features')
    
    args = parser.parse_args()
    
    engineer = FeatureEngineer(args.db_url, args.output_dir)
    
    if not engineer.connect():
        return 1
    
    features, features_scaled, feature_cols = engineer.run_pipeline()
    
    print("\n" + "=" * 60)
    print("FILES GENERATED:")
    print("=" * 60)
    print(f"  customer_features.csv       - Raw features ({len(features):,} customers)")
    print(f"  customer_features_scaled.csv - Scaled features for clustering")
    print(f"  feature_columns.txt         - List of clustering features ({len(feature_cols)} features)")
    print(f"  transformers.pkl            - Fitted transformers for production")
    print(f"  feature_summary.csv         - Feature statistics")
    print(f"  feature_documentation.md    - Feature documentation")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
