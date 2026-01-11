#!/usr/bin/env python3
"""
DATA EXPLORATION & PROFILING
============================
Comprehensive exploratory data analysis for the Olist dataset.
This script generates insights that inform feature engineering decisions.

Usage:
    python data_exploration.py --db-url postgresql://... --output-dir ./reports

Author: Marketing AI Platform
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine, text

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class OlistDataExplorer:
    """
    Comprehensive data exploration for the Olist dataset.
    Generates insights for feature engineering and clustering.
    """
    
    def __init__(self, db_url: str, output_dir: str = './exploration_reports'):
        """
        Initialize the explorer.
        
        Args:
            db_url: PostgreSQL connection string
            output_dir: Directory to save reports and visualizations
        """
        self.db_url = db_url
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.engine = None
        self.data = {}
        
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
    
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load all relevant tables into memory."""
        logger.info("Loading data from database...")
        
        queries = {
            'customers': """
                SELECT * FROM olist.customers
            """,
            'orders': """
                SELECT * FROM olist.orders
            """,
            'order_items': """
                SELECT * FROM olist.order_items
            """,
            'order_payments': """
                SELECT * FROM olist.order_payments
            """,
            'order_reviews': """
                SELECT * FROM olist.order_reviews
            """,
            'products': """
                SELECT p.*, pct.product_category_name_english
                FROM olist.products p
                LEFT JOIN olist.product_category_translation pct
                    ON p.product_category_name = pct.product_category_name
            """,
            'sellers': """
                SELECT * FROM olist.sellers
            """
        }
        
        for name, query in queries.items():
            logger.info(f"  Loading {name}...")
            self.data[name] = pd.read_sql(query, self.engine)
            logger.info(f"    Loaded {len(self.data[name]):,} rows")
        
        return self.data
    
    # =========================================================================
    # BASIC STATISTICS
    # =========================================================================
    
    def generate_basic_stats(self) -> pd.DataFrame:
        """Generate basic statistics for all tables."""
        stats = []
        
        for name, df in self.data.items():
            stats.append({
                'table': name,
                'rows': len(df),
                'columns': len(df.columns),
                'memory_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
                'missing_cells': df.isnull().sum().sum(),
                'missing_pct': (df.isnull().sum().sum() / df.size) * 100
            })
        
        stats_df = pd.DataFrame(stats)
        logger.info("\n" + stats_df.to_string())
        return stats_df
    
    # =========================================================================
    # CUSTOMER ANALYSIS
    # =========================================================================
    
    def analyze_customers(self) -> Dict:
        """Analyze customer distribution and behavior."""
        logger.info("Analyzing customers...")
        
        customers = self.data['customers']
        orders = self.data['orders']
        
        # Merge to get customer orders
        customer_orders = customers.merge(orders, on='customer_id')
        
        analysis = {
            'total_customers': customers['customer_id'].nunique(),
            'unique_customers': customers['customer_unique_id'].nunique(),
            'repeat_customers': customers['customer_unique_id'].value_counts(),
            'customers_by_state': customers['customer_state'].value_counts(),
            'customers_by_city': customers['customer_city'].value_counts().head(20)
        }
        
        # Repeat purchase analysis
        orders_per_customer = customer_orders.groupby('customer_unique_id').size()
        analysis['orders_per_customer_stats'] = orders_per_customer.describe()
        analysis['repeat_purchase_rate'] = (orders_per_customer > 1).mean() * 100
        
        # Visualization: Customer distribution by state
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Top 10 states
        top_states = analysis['customers_by_state'].head(10)
        axes[0].barh(top_states.index[::-1], top_states.values[::-1])
        axes[0].set_xlabel('Number of Customers')
        axes[0].set_title('Top 10 States by Customer Count')
        
        # Orders per customer distribution
        axes[1].hist(orders_per_customer.clip(upper=10), bins=10, edgecolor='black')
        axes[1].set_xlabel('Number of Orders')
        axes[1].set_ylabel('Number of Customers')
        axes[1].set_title(f'Orders per Customer Distribution\n(Repeat Rate: {analysis["repeat_purchase_rate"]:.1f}%)')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'customer_analysis.png', dpi=150)
        plt.close()
        
        logger.info(f"  Total customers: {analysis['total_customers']:,}")
        logger.info(f"  Unique customers: {analysis['unique_customers']:,}")
        logger.info(f"  Repeat purchase rate: {analysis['repeat_purchase_rate']:.2f}%")
        
        return analysis
    
    # =========================================================================
    # ORDER ANALYSIS
    # =========================================================================
    
    def analyze_orders(self) -> Dict:
        """Analyze order patterns and trends."""
        logger.info("Analyzing orders...")
        
        orders = self.data['orders'].copy()
        order_items = self.data['order_items'].copy()
        
        # Convert timestamps
        orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
        orders['order_delivered_customer_date'] = pd.to_datetime(orders['order_delivered_customer_date'])
        
        analysis = {
            'total_orders': len(orders),
            'order_status_distribution': orders['order_status'].value_counts(),
            'date_range': (orders['order_purchase_timestamp'].min(), 
                          orders['order_purchase_timestamp'].max())
        }
        
        # Order value analysis
        order_values = order_items.groupby('order_id').agg({
            'price': 'sum',
            'freight_value': 'sum',
            'order_item_id': 'count'
        }).rename(columns={'order_item_id': 'item_count'})
        order_values['total_value'] = order_values['price'] + order_values['freight_value']
        
        analysis['order_value_stats'] = order_values['total_value'].describe()
        analysis['items_per_order_stats'] = order_values['item_count'].describe()
        
        # Time-based analysis
        orders['order_month'] = orders['order_purchase_timestamp'].dt.to_period('M')
        orders['order_dow'] = orders['order_purchase_timestamp'].dt.dayofweek
        orders['order_hour'] = orders['order_purchase_timestamp'].dt.hour
        
        analysis['orders_by_month'] = orders.groupby('order_month').size()
        analysis['orders_by_dow'] = orders.groupby('order_dow').size()
        analysis['orders_by_hour'] = orders.groupby('order_hour').size()
        
        # Delivery time analysis (for delivered orders)
        delivered = orders[orders['order_status'] == 'delivered'].copy()
        delivered['delivery_days'] = (
            delivered['order_delivered_customer_date'] - 
            delivered['order_purchase_timestamp']
        ).dt.days
        
        analysis['delivery_time_stats'] = delivered['delivery_days'].describe()
        
        # Visualizations
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Order trend over time
        monthly_orders = analysis['orders_by_month']
        axes[0, 0].plot(range(len(monthly_orders)), monthly_orders.values, marker='o')
        axes[0, 0].set_xlabel('Month')
        axes[0, 0].set_ylabel('Number of Orders')
        axes[0, 0].set_title('Monthly Order Trend')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Order value distribution
        axes[0, 1].hist(order_values['total_value'].clip(upper=500), bins=50, edgecolor='black')
        axes[0, 1].set_xlabel('Order Value (BRL)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Order Value Distribution (clipped at 500)')
        
        # Orders by day of week
        dow_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        axes[1, 0].bar(dow_labels, analysis['orders_by_dow'].values)
        axes[1, 0].set_xlabel('Day of Week')
        axes[1, 0].set_ylabel('Number of Orders')
        axes[1, 0].set_title('Orders by Day of Week')
        
        # Order status distribution
        status_counts = analysis['order_status_distribution']
        axes[1, 1].pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%')
        axes[1, 1].set_title('Order Status Distribution')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'order_analysis.png', dpi=150)
        plt.close()
        
        logger.info(f"  Total orders: {analysis['total_orders']:,}")
        logger.info(f"  Average order value: {analysis['order_value_stats']['mean']:.2f} BRL")
        logger.info(f"  Average items per order: {analysis['items_per_order_stats']['mean']:.2f}")
        
        return analysis
    
    # =========================================================================
    # PRODUCT ANALYSIS
    # =========================================================================
    
    def analyze_products(self) -> Dict:
        """Analyze product catalog and categories."""
        logger.info("Analyzing products...")
        
        products = self.data['products']
        order_items = self.data['order_items']
        
        analysis = {
            'total_products': len(products),
            'categories': products['product_category_name_english'].nunique(),
            'category_distribution': products['product_category_name_english'].value_counts()
        }
        
        # Product sales analysis
        product_sales = order_items.groupby('product_id').agg({
            'order_id': 'count',
            'price': ['sum', 'mean']
        })
        product_sales.columns = ['order_count', 'total_revenue', 'avg_price']
        product_sales = product_sales.reset_index()
        
        # Merge with product info
        product_performance = products.merge(product_sales, on='product_id', how='left')
        product_performance['order_count'] = product_performance['order_count'].fillna(0)
        
        # Category performance
        category_performance = product_performance.groupby('product_category_name_english').agg({
            'product_id': 'count',
            'order_count': 'sum',
            'total_revenue': 'sum'
        }).rename(columns={'product_id': 'product_count'})
        category_performance = category_performance.sort_values('total_revenue', ascending=False)
        
        analysis['category_performance'] = category_performance
        analysis['product_sales_stats'] = product_sales['order_count'].describe()
        
        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Top 15 categories by revenue
        top_categories = category_performance.head(15)
        axes[0].barh(top_categories.index[::-1], top_categories['total_revenue'].values[::-1])
        axes[0].set_xlabel('Total Revenue (BRL)')
        axes[0].set_title('Top 15 Categories by Revenue')
        
        # Products per category distribution
        products_per_cat = analysis['category_distribution'].head(15)
        axes[1].barh(products_per_cat.index[::-1], products_per_cat.values[::-1])
        axes[1].set_xlabel('Number of Products')
        axes[1].set_title('Top 15 Categories by Product Count')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'product_analysis.png', dpi=150)
        plt.close()
        
        logger.info(f"  Total products: {analysis['total_products']:,}")
        logger.info(f"  Categories: {analysis['categories']}")
        
        return analysis
    
    # =========================================================================
    # REVIEW ANALYSIS
    # =========================================================================
    
    def analyze_reviews(self) -> Dict:
        """Analyze customer reviews and ratings."""
        logger.info("Analyzing reviews...")
        
        reviews = self.data['order_reviews'].copy()
        
        analysis = {
            'total_reviews': len(reviews),
            'reviews_with_text': reviews['review_comment_message'].notna().sum(),
            'rating_distribution': reviews['review_score'].value_counts().sort_index()
        }
        
        # Rating statistics
        analysis['rating_stats'] = reviews['review_score'].describe()
        analysis['avg_rating'] = reviews['review_score'].mean()
        
        # Text length analysis (for reviews with text)
        reviews_with_text = reviews[reviews['review_comment_message'].notna()].copy()
        reviews_with_text['text_length'] = reviews_with_text['review_comment_message'].str.len()
        analysis['text_length_stats'] = reviews_with_text['text_length'].describe()
        
        # Rating by text presence
        analysis['avg_rating_with_text'] = reviews_with_text['review_score'].mean()
        analysis['avg_rating_without_text'] = reviews[reviews['review_comment_message'].isna()]['review_score'].mean()
        
        # Visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Rating distribution
        rating_dist = analysis['rating_distribution']
        colors = ['#d73027', '#fc8d59', '#fee090', '#91cf60', '#1a9850']
        axes[0].bar(rating_dist.index, rating_dist.values, color=colors)
        axes[0].set_xlabel('Rating')
        axes[0].set_ylabel('Count')
        axes[0].set_title(f'Rating Distribution\n(Avg: {analysis["avg_rating"]:.2f})')
        
        # Reviews with/without text
        text_counts = [analysis['reviews_with_text'], 
                      analysis['total_reviews'] - analysis['reviews_with_text']]
        axes[1].pie(text_counts, labels=['With Text', 'Without Text'], 
                   autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'])
        axes[1].set_title('Reviews with Text Comments')
        
        # Text length distribution
        if len(reviews_with_text) > 0:
            axes[2].hist(reviews_with_text['text_length'].clip(upper=500), 
                        bins=50, edgecolor='black')
            axes[2].set_xlabel('Text Length (characters)')
            axes[2].set_ylabel('Frequency')
            axes[2].set_title('Review Text Length Distribution')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'review_analysis.png', dpi=150)
        plt.close()
        
        logger.info(f"  Total reviews: {analysis['total_reviews']:,}")
        logger.info(f"  Reviews with text: {analysis['reviews_with_text']:,} ({analysis['reviews_with_text']/analysis['total_reviews']*100:.1f}%)")
        logger.info(f"  Average rating: {analysis['avg_rating']:.2f}")
        
        return analysis
    
    # =========================================================================
    # PAYMENT ANALYSIS
    # =========================================================================
    
    def analyze_payments(self) -> Dict:
        """Analyze payment methods and patterns."""
        logger.info("Analyzing payments...")
        
        payments = self.data['order_payments']
        
        analysis = {
            'total_payments': len(payments),
            'payment_types': payments['payment_type'].value_counts(),
            'installments_dist': payments['payment_installments'].value_counts().sort_index()
        }
        
        # Payment value by type
        payment_by_type = payments.groupby('payment_type').agg({
            'payment_value': ['sum', 'mean', 'count']
        })
        payment_by_type.columns = ['total_value', 'avg_value', 'count']
        analysis['payment_by_type'] = payment_by_type
        
        # Installment analysis (credit card only)
        credit_card = payments[payments['payment_type'] == 'credit_card']
        analysis['credit_card_installments'] = credit_card['payment_installments'].describe()
        
        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Payment type distribution
        type_counts = analysis['payment_types']
        axes[0].pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%')
        axes[0].set_title('Payment Type Distribution')
        
        # Installments distribution (for credit card)
        inst_dist = credit_card['payment_installments'].value_counts().sort_index().head(12)
        axes[1].bar(inst_dist.index, inst_dist.values)
        axes[1].set_xlabel('Number of Installments')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Credit Card Installments Distribution')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'payment_analysis.png', dpi=150)
        plt.close()
        
        logger.info(f"  Total payments: {analysis['total_payments']:,}")
        logger.info(f"  Most common payment: {analysis['payment_types'].index[0]}")
        
        return analysis
    
    # =========================================================================
    # RFM ANALYSIS (Key for Clustering)
    # =========================================================================
    
    def analyze_rfm(self) -> Tuple[pd.DataFrame, Dict]:
        """
        Perform RFM (Recency, Frequency, Monetary) analysis.
        This is the foundation for customer segmentation.
        """
        logger.info("Performing RFM analysis...")
        
        # Get customer orders with values
        customers = self.data['customers']
        orders = self.data['orders'].copy()
        order_items = self.data['order_items']
        
        # Calculate order values
        order_values = order_items.groupby('order_id').agg({
            'price': 'sum',
            'freight_value': 'sum'
        })
        order_values['total_value'] = order_values['price'] + order_values['freight_value']
        
        # Merge orders with values and customers
        orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
        orders = orders.merge(order_values[['total_value']], left_on='order_id', right_index=True)
        orders = orders.merge(customers[['customer_id', 'customer_unique_id']], on='customer_id')
        
        # Calculate RFM metrics
        # Reference date = max date in dataset + 1 day
        reference_date = orders['order_purchase_timestamp'].max() + pd.Timedelta(days=1)
        
        rfm = orders.groupby('customer_unique_id').agg({
            'order_purchase_timestamp': lambda x: (reference_date - x.max()).days,  # Recency
            'order_id': 'nunique',  # Frequency
            'total_value': 'sum'  # Monetary
        }).rename(columns={
            'order_purchase_timestamp': 'recency',
            'order_id': 'frequency',
            'total_value': 'monetary'
        })
        
        # Calculate additional metrics
        rfm['avg_order_value'] = rfm['monetary'] / rfm['frequency']
        
        # RFM Statistics
        rfm_stats = {
            'recency': rfm['recency'].describe(),
            'frequency': rfm['frequency'].describe(),
            'monetary': rfm['monetary'].describe(),
            'avg_order_value': rfm['avg_order_value'].describe()
        }
        
        # RFM Score calculation (quintiles)
        rfm['R_score'] = pd.qcut(rfm['recency'], q=5, labels=[5, 4, 3, 2, 1], duplicates='drop')
        rfm['F_score'] = pd.qcut(rfm['frequency'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')
        rfm['M_score'] = pd.qcut(rfm['monetary'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')
        
        rfm['RFM_score'] = rfm['R_score'].astype(str) + rfm['F_score'].astype(str) + rfm['M_score'].astype(str)
        
        # Segment mapping based on RFM scores
        def segment_customer(row):
            r, f, m = int(row['R_score']), int(row['F_score']), int(row['M_score'])
            
            if r >= 4 and f >= 4 and m >= 4:
                return 'Champions'
            elif r >= 4 and f >= 3:
                return 'Loyal Customers'
            elif r >= 4 and f <= 2:
                return 'New Customers'
            elif r >= 3 and f >= 3 and m >= 3:
                return 'Potential Loyalists'
            elif r <= 2 and f >= 4:
                return 'At Risk'
            elif r <= 2 and f <= 2 and m <= 2:
                return 'Lost'
            elif r <= 2 and f >= 2:
                return 'Hibernating'
            else:
                return 'Need Attention'
        
        rfm['segment'] = rfm.apply(segment_customer, axis=1)
        rfm_stats['segment_distribution'] = rfm['segment'].value_counts()
        
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Recency distribution
        axes[0, 0].hist(rfm['recency'].clip(upper=365), bins=50, edgecolor='black')
        axes[0, 0].set_xlabel('Days Since Last Purchase')
        axes[0, 0].set_ylabel('Number of Customers')
        axes[0, 0].set_title('Recency Distribution')
        axes[0, 0].axvline(rfm['recency'].median(), color='red', linestyle='--', label=f'Median: {rfm["recency"].median():.0f}')
        axes[0, 0].legend()
        
        # Frequency distribution
        axes[0, 1].hist(rfm['frequency'].clip(upper=10), bins=10, edgecolor='black')
        axes[0, 1].set_xlabel('Number of Orders')
        axes[0, 1].set_ylabel('Number of Customers')
        axes[0, 1].set_title('Frequency Distribution')
        
        # Monetary distribution
        axes[1, 0].hist(rfm['monetary'].clip(upper=2000), bins=50, edgecolor='black')
        axes[1, 0].set_xlabel('Total Spend (BRL)')
        axes[1, 0].set_ylabel('Number of Customers')
        axes[1, 0].set_title('Monetary Distribution')
        
        # Segment distribution
        segment_counts = rfm_stats['segment_distribution']
        colors = plt.cm.Set3(np.linspace(0, 1, len(segment_counts)))
        axes[1, 1].barh(segment_counts.index, segment_counts.values, color=colors)
        axes[1, 1].set_xlabel('Number of Customers')
        axes[1, 1].set_title('RFM Segment Distribution')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'rfm_analysis.png', dpi=150)
        plt.close()
        
        # RFM Heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        rfm_heatmap = rfm.groupby(['R_score', 'F_score']).size().unstack(fill_value=0)
        sns.heatmap(rfm_heatmap, annot=True, fmt='d', cmap='YlGnBu', ax=ax)
        ax.set_xlabel('Frequency Score')
        ax.set_ylabel('Recency Score')
        ax.set_title('RFM Heatmap (Customer Count)')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'rfm_heatmap.png', dpi=150)
        plt.close()
        
        logger.info(f"  RFM analysis complete for {len(rfm):,} customers")
        logger.info(f"  Segments:\n{rfm_stats['segment_distribution']}")
        
        # Save RFM data for later use
        rfm.to_csv(self.output_dir / 'rfm_data.csv')
        
        return rfm, rfm_stats
    
    # =========================================================================
    # CORRELATION ANALYSIS
    # =========================================================================
    
    def analyze_correlations(self, rfm: pd.DataFrame) -> pd.DataFrame:
        """Analyze correlations between RFM metrics."""
        logger.info("Analyzing correlations...")
        
        # Numeric columns for correlation
        numeric_cols = ['recency', 'frequency', 'monetary', 'avg_order_value']
        corr_matrix = rfm[numeric_cols].corr()
        
        # Visualization
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   fmt='.2f', ax=ax, vmin=-1, vmax=1)
        ax.set_title('RFM Metrics Correlation Matrix')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'correlation_matrix.png', dpi=150)
        plt.close()
        
        return corr_matrix
    
    # =========================================================================
    # GENERATE FULL REPORT
    # =========================================================================
    
    def generate_report(self) -> str:
        """Generate a comprehensive exploration report."""
        logger.info("Generating exploration report...")
        
        # Run all analyses
        basic_stats = self.generate_basic_stats()
        customer_analysis = self.analyze_customers()
        order_analysis = self.analyze_orders()
        product_analysis = self.analyze_products()
        review_analysis = self.analyze_reviews()
        payment_analysis = self.analyze_payments()
        rfm_data, rfm_stats = self.analyze_rfm()
        corr_matrix = self.analyze_correlations(rfm_data)
        
        # Generate markdown report
        report = f"""# Olist Data Exploration Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This report provides a comprehensive exploration of the Olist Brazilian E-Commerce dataset
to inform feature engineering for customer segmentation.

## Dataset Overview

{basic_stats.to_markdown()}

## Customer Analysis

- **Total Customer IDs**: {customer_analysis['total_customers']:,}
- **Unique Customers**: {customer_analysis['unique_customers']:,}
- **Repeat Purchase Rate**: {customer_analysis['repeat_purchase_rate']:.2f}%

### Orders per Customer
{customer_analysis['orders_per_customer_stats'].to_frame().to_markdown()}

### Top States by Customer Count
{customer_analysis['customers_by_state'].head(10).to_frame().to_markdown()}

## Order Analysis

- **Total Orders**: {order_analysis['total_orders']:,}
- **Date Range**: {order_analysis['date_range'][0]} to {order_analysis['date_range'][1]}

### Order Value Statistics
{order_analysis['order_value_stats'].to_frame().to_markdown()}

### Order Status Distribution
{order_analysis['order_status_distribution'].to_frame().to_markdown()}

### Delivery Time (days)
{order_analysis['delivery_time_stats'].to_frame().to_markdown()}

## Product Analysis

- **Total Products**: {product_analysis['total_products']:,}
- **Categories**: {product_analysis['categories']}

### Top Categories by Revenue
{product_analysis['category_performance'].head(10).to_markdown()}

## Review Analysis

- **Total Reviews**: {review_analysis['total_reviews']:,}
- **Reviews with Text**: {review_analysis['reviews_with_text']:,} ({review_analysis['reviews_with_text']/review_analysis['total_reviews']*100:.1f}%)
- **Average Rating**: {review_analysis['avg_rating']:.2f}

### Rating Distribution
{review_analysis['rating_distribution'].to_frame().to_markdown()}

## Payment Analysis

### Payment Type Distribution
{payment_analysis['payment_types'].to_frame().to_markdown()}

## RFM Analysis

### RFM Statistics

**Recency (days since last purchase)**
{rfm_stats['recency'].to_frame().to_markdown()}

**Frequency (number of orders)**
{rfm_stats['frequency'].to_frame().to_markdown()}

**Monetary (total spend in BRL)**
{rfm_stats['monetary'].to_frame().to_markdown()}

### Customer Segments
{rfm_stats['segment_distribution'].to_frame().to_markdown()}

## Correlation Analysis

{corr_matrix.to_markdown()}

## Key Insights for Clustering

1. **Low Repeat Purchase Rate** ({customer_analysis['repeat_purchase_rate']:.2f}%): Most customers are one-time buyers, 
   which affects frequency distribution.

2. **High Variance in Monetary**: Large spread in customer spending suggests clear segments.

3. **Delivery Experience Matters**: Average delivery of {order_analysis['delivery_time_stats']['mean']:.1f} days 
   may influence satisfaction.

4. **Review Engagement**: Only {review_analysis['reviews_with_text']/review_analysis['total_reviews']*100:.1f}% 
   leave text reviews, but these are valuable for sentiment analysis.

5. **Geographic Concentration**: {customer_analysis['customers_by_state'].index[0]} dominates 
   with {customer_analysis['customers_by_state'].iloc[0]:,} customers.

## Recommended Features for Clustering

Based on this analysis, the following features are recommended for customer clustering:

### Core RFM Features
- Recency (days since last purchase)
- Frequency (order count)
- Monetary (total spend)
- Average order value

### Behavioral Features
- Number of product categories purchased
- Favorite category
- Payment method preference
- Average installments used

### Engagement Features
- Review engagement rate
- Average review score given
- Review text length (if provided)

### Geographic Features
- Customer state (encoded)
- Distance from sellers (if using geolocation)

## Next Steps

1. Run feature engineering pipeline (02_feature_engineering.py)
2. Apply scaling and transformation
3. Run K-Means and HDBSCAN clustering
4. Evaluate and compare clusters
5. Generate cluster profiles for marketing
"""
        
        # Save report
        report_path = self.output_dir / 'exploration_report.md'
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Report saved to {report_path}")
        
        return report


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Olist Data Exploration')
    parser.add_argument('--db-url', type=str, required=True,
                       help='PostgreSQL connection URL')
    parser.add_argument('--output-dir', type=str, default='./exploration_reports',
                       help='Output directory for reports')
    
    args = parser.parse_args()
    
    explorer = OlistDataExplorer(args.db_url, args.output_dir)
    
    if not explorer.connect():
        return 1
    
    explorer.load_data()
    report = explorer.generate_report()
    
    print("\n" + "=" * 60)
    print("EXPLORATION COMPLETE")
    print("=" * 60)
    print(f"Reports saved to: {args.output_dir}")
    print("Files generated:")
    print("  - exploration_report.md")
    print("  - customer_analysis.png")
    print("  - order_analysis.png")
    print("  - product_analysis.png")
    print("  - review_analysis.png")
    print("  - payment_analysis.png")
    print("  - rfm_analysis.png")
    print("  - rfm_heatmap.png")
    print("  - correlation_matrix.png")
    print("  - rfm_data.csv")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
