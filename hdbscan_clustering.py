#!/usr/bin/env python3
"""
HDBSCAN CLUSTERING - MARKETING-FOCUSED SEGMENTATION
=======================================================
This version ensures:
1. NO ambiguous "Core Customers" catch-all bucket
2. Every segment has CLEAR marketing actions
3. Clusters are used for personalization WITHIN cohorts
4. Business-meaningful segments that marketers can act on

Key Principle:
- COHORTS = WHAT campaign to run (strategy)
- CLUSTERS = HOW to personalize it (tactics)

Usage:
    python hdbscan_clustering.py \
        --features-dir ./features \
        --output-dir ./clustering_results

"""

import os
import argparse
import logging
import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import pandas as pd
import numpy as np
import hdbscan
from sklearn.metrics import silhouette_score
import warnings

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================
# MARKETING-FOCUSED SEGMENT DEFINITIONS
# ============================================================
"""
RFM Score Ranges:
- R (Recency): 5=recent, 1=long ago
- F (Frequency): 5=frequent, 1=rare
- M (Monetary): 5=high value, 1=low value

Each segment has:
1. Clear definition
2. Specific marketing action
3. Priority score
4. KPIs to track
"""

SEGMENT_DEFINITIONS = {
    'Champions': {
        'description': 'Best customers - recent, frequent, high-value buyers',
        'rfm_condition': lambda r, f, m: r >= 4 and f >= 4 and m >= 4,
        'priority': 10,
        'marketing_goal': 'Retain and grow',
        'actions': [
            'VIP loyalty program enrollment',
            'Early access to new products',
            'Exclusive discounts (15-20%)',
            'Personal account manager',
            'Referral program with premium rewards'
        ],
        'kpis': ['Retention rate', 'Lifetime value growth', 'Referral rate'],
        'budget_allocation': '20%'
    },
    'Loyal Customers': {
        'description': 'Regular buyers with strong engagement',
        'rfm_condition': lambda r, f, m: f >= 4 and r >= 3,
        'priority': 9,
        'marketing_goal': 'Increase basket size',
        'actions': [
            'Cross-sell recommendations',
            'Bundle offers',
            'Loyalty points program',
            'Birthday/anniversary rewards',
            'Category expansion campaigns'
        ],
        'kpis': ['Average order value', 'Items per order', 'Category penetration'],
        'budget_allocation': '15%'
    },
    'Potential Loyalists': {
        'description': 'Recent buyers showing promise - need nurturing',
        'rfm_condition': lambda r, f, m: r >= 4 and f >= 2 and f < 4 and m >= 2,
        'priority': 8,
        'marketing_goal': 'Convert to loyal',
        'actions': [
            'Onboarding email series',
            'Second/third purchase incentive',
            'Product education content',
            'Community invitation',
            'Feedback collection'
        ],
        'kpis': ['Repeat purchase rate', 'Time to second purchase', 'Engagement rate'],
        'budget_allocation': '15%'
    },
    'Recent Customers': {
        'description': 'Just made first purchase - critical nurturing window',
        'rfm_condition': lambda r, f, m: r >= 4 and f == 1,
        'priority': 7,
        'marketing_goal': 'Drive second purchase',
        'actions': [
            'Welcome email sequence',
            'How-to guides for purchased products',
            'Complementary product suggestions',
            '10% off next purchase',
            'Mobile app download incentive'
        ],
        'kpis': ['30-day repeat rate', 'Email engagement', 'App adoption'],
        'budget_allocation': '10%'
    },
    'Promising': {
        'description': 'Moderate engagement with growth potential',
        'rfm_condition': lambda r, f, m: r >= 3 and f >= 2 and m >= 3,
        'priority': 6,
        'marketing_goal': 'Increase engagement',
        'actions': [
            'Personalized recommendations',
            'Limited-time offers',
            'Gamification program',
            'Social proof campaigns',
            'User-generated content requests'
        ],
        'kpis': ['Purchase frequency', 'Email click rate', 'Site visits'],
        'budget_allocation': '10%'
    },
    'Need Attention': {
        'description': 'Above average but slowing down - intervention needed',
        'rfm_condition': lambda r, f, m: r == 3 and f >= 3 and m >= 3,
        'priority': 7,
        'marketing_goal': 'Re-engage before decline',
        'actions': [
            'Personal check-in email',
            '"We miss you" campaign',
            'Exclusive comeback offer',
            'Feedback survey',
            'New product announcements'
        ],
        'kpis': ['Response rate', 'Return rate', 'Feedback completion'],
        'budget_allocation': '8%'
    },
    'About to Sleep': {
        'description': 'Declining engagement - last chance for retention',
        'rfm_condition': lambda r, f, m: r == 2 and f >= 2,
        'priority': 8,
        'marketing_goal': 'Prevent churn',
        'actions': [
            'Urgent reactivation campaign',
            'Strong discount (25-30%)',
            '"Last chance" messaging',
            'Product updates showcase',
            'Easy return/exchange reminder'
        ],
        'kpis': ['Reactivation rate', 'Days to next purchase', 'Offer redemption'],
        'budget_allocation': '7%'
    },
    'At Risk': {
        'description': 'Were valuable, now disengaging - high priority save',
        'rfm_condition': lambda r, f, m: r <= 2 and (f >= 3 or m >= 4),
        'priority': 9,
        'marketing_goal': 'Win back immediately',
        'actions': [
            'Aggressive win-back offer (30-40%)',
            'Personal outreach call/email',
            '"We want you back" campaign',
            'Competitor comparison content',
            'Free shipping for 3 months'
        ],
        'kpis': ['Win-back rate', 'Revenue recovered', 'Response rate'],
        'budget_allocation': '5%'
    },
    'Hibernating': {
        'description': 'Low recent activity but have purchase history',
        'rfm_condition': lambda r, f, m: r <= 2 and f <= 2 and m >= 2,
        'priority': 4,
        'marketing_goal': 'Reawaken interest',
        'actions': [
            'Re-engagement email series',
            'Major sale announcements',
            'New category introductions',
            'Brand story content',
            'Social media retargeting'
        ],
        'kpis': ['Email open rate', 'Site return rate', 'Offer redemption'],
        'budget_allocation': '3%'
    },
    'Lost': {
        'description': 'No activity for extended period - lowest priority',
        'rfm_condition': lambda r, f, m: r == 1 and f <= 2 and m <= 2,
        'priority': 2,
        'marketing_goal': 'Last attempt or sunset',
        'actions': [
            'Final win-back attempt (one email)',
            'Exit survey invitation',
            'Account deactivation notice',
            'Referral request',
            'Remove from active campaigns'
        ],
        'kpis': ['Final response rate', 'Survey completion', 'Referral conversion'],
        'budget_allocation': '2%'
    },
    'Price Sensitive': {
        'description': 'Buy mainly during sales - value seekers',
        'rfm_condition': lambda r, f, m: f >= 2 and m <= 2 and r >= 2,
        'priority': 5,
        'marketing_goal': 'Increase value per transaction',
        'actions': [
            'Bundle deals',
            'Bulk purchase discounts',
            'Clearance alerts',
            'Price drop notifications',
            'Value-focused messaging'
        ],
        'kpis': ['Average order value', 'Sale vs regular purchases', 'Bundle adoption'],
        'budget_allocation': '5%'
    }
}

# Evaluation order matters! More specific conditions first
SEGMENT_PRIORITY_ORDER = [
    'Champions',
    'Loyal Customers', 
    'At Risk',
    'About to Sleep',
    'Potential Loyalists',
    'Recent Customers',
    'Need Attention',
    'Promising',
    'Price Sensitive',
    'Hibernating',
    'Lost'
]


def assign_marketing_segment(r: int, f: int, m: int) -> str:
    """
    Assign customer to marketing segment based on RFM scores.
    Evaluates conditions in priority order - no catch-all bucket!
    """
    for segment_name in SEGMENT_PRIORITY_ORDER:
        segment = SEGMENT_DEFINITIONS[segment_name]
        if segment['rfm_condition'](r, f, m):
            return segment_name
    
    # Fallback based on recency (should rarely hit this)
    if r >= 3:
        return 'Promising'
    elif r == 2:
        return 'Hibernating'
    else:
        return 'Lost'


# ============================================================
# RFM SCORING
# ============================================================

def calculate_rfm_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate RFM quintile scores (1-5) for each customer.
    5 = best, 1 = worst
    """
    df = df.copy()
    
    # Recency: Lower days = better = higher score
    # Use negative to reverse the ranking
    df['R_score'] = pd.qcut(
        -df['recency'].rank(method='first'),  # Negative to reverse
        q=5, 
        labels=[1, 2, 3, 4, 5]
    ).astype(int)
    
    # Frequency: Higher = better
    df['F_score'] = pd.qcut(
        df['frequency'].rank(method='first'), 
        q=5, 
        labels=[1, 2, 3, 4, 5],
        duplicates='drop'
    ).astype(int)
    
    # Monetary: Higher = better
    df['M_score'] = pd.qcut(
        df['monetary'].rank(method='first'), 
        q=5, 
        labels=[1, 2, 3, 4, 5],
        duplicates='drop'
    ).astype(int)
    
    # Combined score
    df['RFM_score'] = df['R_score'].astype(str) + df['F_score'].astype(str) + df['M_score'].astype(str)
    df['RFM_total'] = df['R_score'] + df['F_score'] + df['M_score']
    
    return df


# ============================================================
# HDBSCAN CLUSTERING
# ============================================================

def fit_hdbscan_with_tuning(
    X: np.ndarray,
    target_noise_pct: float = 0.15
) -> Tuple[hdbscan.HDBSCAN, np.ndarray, np.ndarray]:
    """
    Fit HDBSCAN with automatic parameter tuning.
    """
    logger.info("Fitting HDBSCAN with parameter tuning...")
    
    best_model = None
    best_labels = None
    best_probs = None
    best_score = -np.inf
    best_params = {}
    
    # Parameter grid
    param_grid = [
        {'min_cluster_size': 30, 'min_samples': 5},
        {'min_cluster_size': 50, 'min_samples': 10},
        {'min_cluster_size': 75, 'min_samples': 15},
        {'min_cluster_size': 100, 'min_samples': 20},
        {'min_cluster_size': 50, 'min_samples': 5},
        {'min_cluster_size': 75, 'min_samples': 10},
    ]
    
    for params in param_grid:
        model = hdbscan.HDBSCAN(
            min_cluster_size=params['min_cluster_size'],
            min_samples=params['min_samples'],
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True
        )
        
        labels = model.fit_predict(X)
        probs = model.probabilities_
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_pct = (labels == -1).sum() / len(labels)
        
        # Score: prefer lower noise and more clusters
        score = (1 - noise_pct) * 0.6 + min(n_clusters / 50, 1) * 0.4
        
        logger.info(f"  params={params}: {n_clusters} clusters, {noise_pct:.1%} noise, score={score:.3f}")
        
        if score > best_score:
            best_score = score
            best_model = model
            best_labels = labels
            best_probs = probs
            best_params = params
    
    logger.info(f"\nBest: {best_params}")
    n_clusters = len(set(best_labels)) - (1 if -1 in best_labels else 0)
    noise_pct = (best_labels == -1).sum() / len(best_labels)
    logger.info(f"  → {n_clusters} clusters, {noise_pct:.1%} noise")
    
    return best_model, best_labels, best_probs


# ============================================================
# CLUSTER PROFILING FOR PERSONALIZATION
# ============================================================

def profile_clusters_for_personalization(
    df: pd.DataFrame,
    feature_cols: List[str]
) -> pd.DataFrame:
    """
    Create cluster profiles to guide personalization within cohorts.
    
    Example use case:
    - Cohort: "Champions" 
    - Cluster A: Champions who buy electronics → Tech-focused messaging
    - Cluster B: Champions who buy fashion → Style-focused messaging
    """
    logger.info("Profiling clusters for personalization...")
    
    cluster_profiles = []
    
    for cluster_id in df['hdbscan_cluster'].unique():
        if cluster_id == -1:
            continue
            
        cluster_df = df[df['hdbscan_cluster'] == cluster_id]
        
        profile = {
            'cluster_id': int(cluster_id),
            'size': len(cluster_df),
            'primary_cohort': cluster_df['cohort_name'].mode().iloc[0],
            'cohort_distribution': cluster_df['cohort_name'].value_counts().to_dict(),
            
            # RFM averages
            'avg_recency': round(cluster_df['recency'].mean(), 1),
            'avg_frequency': round(cluster_df['frequency'].mean(), 2),
            'avg_monetary': round(cluster_df['monetary'].mean(), 2),
            'avg_rfm_total': round(cluster_df['RFM_total'].mean(), 2),
        }
        
        # Add category preferences if available
        if 'top_category' in cluster_df.columns:
            top_cats = cluster_df['top_category'].value_counts().head(3)
            profile['top_categories'] = top_cats.to_dict()
            profile['primary_category'] = top_cats.index[0] if len(top_cats) > 0 else 'various'
        
        # Add state distribution if available
        if 'customer_state' in cluster_df.columns:
            top_states = cluster_df['customer_state'].value_counts().head(3)
            profile['top_states'] = top_states.to_dict()
        
        # Personalization recommendation
        profile['personalization_hint'] = generate_personalization_hint(profile)
        
        cluster_profiles.append(profile)
    
    return pd.DataFrame(cluster_profiles)


def generate_personalization_hint(profile: dict) -> str:
    """Generate a personalization recommendation based on cluster profile."""
    hints = []
    
    # Category-based
    if 'primary_category' in profile:
        cat = profile['primary_category']
        if cat in ['electronics', 'computers']:
            hints.append("Tech-focused messaging, feature comparisons")
        elif cat in ['fashion', 'clothing']:
            hints.append("Style guides, trend updates, lookbooks")
        elif cat in ['home', 'furniture']:
            hints.append("Home improvement tips, room inspiration")
        elif cat in ['beauty', 'health']:
            hints.append("Tutorial content, routine builders")
    
    # Value-based
    if profile['avg_monetary'] > 500:
        hints.append("Premium product recommendations")
    elif profile['avg_monetary'] < 100:
        hints.append("Value bundles, budget-friendly options")
    
    # Frequency-based
    if profile['avg_frequency'] > 3:
        hints.append("Subscription/auto-replenish offers")
    elif profile['avg_frequency'] == 1:
        hints.append("Discovery content, category expansion")
    
    return "; ".join(hints) if hints else "General personalization"


# ============================================================
# MAIN PIPELINE
# ============================================================

class MarketingClusterer:
    """
    Marketing-focused customer segmentation using HDBSCAN + RFM.
    
    Output structure:
    1. customer_cohorts.csv - Every customer with segment assignment
    2. cohort_profiles.csv - Marketing actions per cohort
    3. cluster_profiles.csv - Personalization guide per cluster
    4. marketing_playbook.json - Complete campaign guide
    """
    
    def __init__(self, features_dir: str, output_dir: str):
        self.features_dir = Path(features_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.features = None
        self.features_scaled = None
        self.feature_cols = None
        self.model = None
    
    def load_features(self) -> bool:
        """Load feature data."""
        logger.info("Loading features...")
        try:
            self.features = pd.read_csv(self.features_dir / 'customer_features.csv')
            self.features_scaled = pd.read_csv(self.features_dir / 'customer_features_scaled.csv')
            
            with open(self.features_dir / 'feature_columns.txt', 'r') as f:
                self.feature_cols = [line.strip() for line in f.readlines()]
            
            logger.info(f"  Loaded {len(self.features):,} customers")
            return True
        except Exception as e:
            logger.error(f"Failed: {e}")
            return False
    
    def run(self):
        """Run complete pipeline."""
        logger.info("=" * 70)
        logger.info("MARKETING-FOCUSED CUSTOMER SEGMENTATION")
        logger.info("=" * 70)
        
        # 1. Load data
        if not self.load_features():
            return False
        
        # 2. Calculate RFM scores
        logger.info("\nCalculating RFM scores...")
        df = calculate_rfm_scores(self.features)
        
        # 3. Fit HDBSCAN
        X = self.features_scaled[self.feature_cols].values
        self.model, labels, probs = fit_hdbscan_with_tuning(X)
        df['hdbscan_cluster'] = labels
        df['cluster_probability'] = probs
        
        # 4. Assign marketing segments (based on RFM, not clusters!)
        logger.info("\nAssigning marketing segments...")
        df['cohort_name'] = df.apply(
            lambda row: assign_marketing_segment(row['R_score'], row['F_score'], row['M_score']),
            axis=1
        )
        df['cohort_id'] = df['cohort_name'].map(
            {name: i+1 for i, name in enumerate(SEGMENT_PRIORITY_ORDER)}
        )
        df['priority'] = df['cohort_name'].map(
            {name: SEGMENT_DEFINITIONS[name]['priority'] for name in SEGMENT_DEFINITIONS}
        )
        
        # 5. Log segment distribution
        logger.info("\n" + "=" * 50)
        logger.info("SEGMENT DISTRIBUTION")
        logger.info("=" * 50)
        
        segment_counts = df['cohort_name'].value_counts()
        for segment in SEGMENT_PRIORITY_ORDER:
            if segment in segment_counts.index:
                count = segment_counts[segment]
                pct = count / len(df) * 100
                priority = SEGMENT_DEFINITIONS[segment]['priority']
                logger.info(f"  {segment}: {count:,} ({pct:.1f}%) [Priority: {priority}]")
        
        # 6. Generate cohort profiles with marketing actions
        logger.info("\nGenerating cohort profiles...")
        cohort_profiles = self._generate_cohort_profiles(df)
        
        # 7. Generate cluster profiles for personalization
        cluster_profiles = profile_clusters_for_personalization(df, self.feature_cols)
        
        # 8. Save everything
        self._save_results(df, cohort_profiles, cluster_profiles)
        
        # 9. Summary
        logger.info("\n" + "=" * 70)
        logger.info("SEGMENTATION COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Total customers: {len(df):,}")
        logger.info(f"Marketing segments: {df['cohort_name'].nunique()}")
        logger.info(f"HDBSCAN clusters: {(df['hdbscan_cluster'] != -1).sum()} in {df['hdbscan_cluster'].nunique()-1} clusters")
        logger.info(f"Noise points: {(df['hdbscan_cluster'] == -1).sum():,} (still assigned to segments via RFM)")
        logger.info("\n✓ Every customer has a marketing segment with clear actions!")
        
        return True
    
    def _generate_cohort_profiles(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate profiles with marketing actions for each cohort."""
        profiles = []
        total = len(df)
        
        for cohort_name in df['cohort_name'].unique():
            cohort_df = df[df['cohort_name'] == cohort_name]
            segment_def = SEGMENT_DEFINITIONS.get(cohort_name, {})
            
            profile = {
                'cohort_id': int(df[df['cohort_name'] == cohort_name]['cohort_id'].iloc[0]),
                'cohort_name': cohort_name,
                'description': segment_def.get('description', ''),
                
                # Size
                'customer_count': len(cohort_df),
                'percentage': round(len(cohort_df) / total * 100, 2),
                
                # RFM stats
                'avg_recency': round(cohort_df['recency'].mean(), 1),
                'avg_frequency': round(cohort_df['frequency'].mean(), 2),
                'avg_monetary': round(cohort_df['monetary'].mean(), 2),
                'total_revenue': round(cohort_df['monetary'].sum(), 2),
                
                # Marketing
                'priority': segment_def.get('priority', 5),
                'marketing_goal': segment_def.get('marketing_goal', ''),
                'recommended_actions': segment_def.get('actions', []),
                'kpis': segment_def.get('kpis', []),
                'budget_allocation': segment_def.get('budget_allocation', ''),
                
                # Clusters
                'cluster_ids': [int(c) for c in cohort_df['hdbscan_cluster'].unique() if c != -1],
                'includes_noise_points': int((cohort_df['hdbscan_cluster'] == -1).sum())
            }
            
            profiles.append(profile)
        
        return pd.DataFrame(profiles).sort_values('priority', ascending=False)
    
    def _save_results(
        self,
        df: pd.DataFrame,
        cohort_profiles: pd.DataFrame,
        cluster_profiles: pd.DataFrame
    ):
        """Save all outputs."""
        logger.info("\nSaving results...")
        
        # 1. Customer cohorts
        output_cols = [
            'customer_unique_id', 'hdbscan_cluster', 'cohort_id', 'cohort_name',
            'priority', 'cluster_probability',
            'recency', 'frequency', 'monetary',
            'R_score', 'F_score', 'M_score', 'RFM_score', 'RFM_total'
        ]
        for col in ['avg_order_value', 'customer_state', 'top_category']:
            if col in df.columns:
                output_cols.append(col)
        
        df[output_cols].to_csv(self.output_dir / 'customer_cohorts.csv', index=False)
        logger.info(f"  ✓ customer_cohorts.csv ({len(df):,} customers)")
        
        # 2. Cohort profiles (with JSON arrays as strings)
        profiles_out = cohort_profiles.copy()
        for col in ['recommended_actions', 'kpis', 'cluster_ids']:
            if col in profiles_out.columns:
                profiles_out[col] = profiles_out[col].apply(json.dumps)
        profiles_out.to_csv(self.output_dir / 'cohort_profiles.csv', index=False)
        logger.info(f"  ✓ cohort_profiles.csv ({len(cohort_profiles)} cohorts)")
        
        # 3. Cluster profiles
        cluster_profiles_out = cluster_profiles.copy()
        for col in ['cohort_distribution', 'top_categories', 'top_states']:
            if col in cluster_profiles_out.columns:
                cluster_profiles_out[col] = cluster_profiles_out[col].apply(
                    lambda x: json.dumps(x) if isinstance(x, dict) else x
                )
        cluster_profiles_out.to_csv(self.output_dir / 'cluster_profiles.csv', index=False)
        logger.info(f"  ✓ cluster_profiles.csv ({len(cluster_profiles)} clusters)")
        
        # 4. Marketing playbook (JSON)
        playbook = {
            'generated_at': pd.Timestamp.now().isoformat(),
            'total_customers': int(len(df)),
            'segments': {}
        }
        
        for _, row in cohort_profiles.iterrows():
            playbook['segments'][row['cohort_name']] = {
                'customer_count': int(row['customer_count']),
                'percentage': float(row['percentage']),
                'description': row['description'],
                'marketing_goal': row['marketing_goal'],
                'priority': int(row['priority']),
                'budget_allocation': row['budget_allocation'],
                'actions': row['recommended_actions'],
                'kpis': row['kpis'],
                'avg_customer_value': float(row['avg_monetary'])
            }
        
        with open(self.output_dir / 'marketing_playbook.json', 'w') as f:
            json.dump(playbook, f, indent=2)
        logger.info("  ✓ marketing_playbook.json")
        
        # 5. Model
        with open(self.output_dir / 'hdbscan_model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        logger.info("  ✓ hdbscan_model.pkl")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Marketing-focused customer segmentation')
    parser.add_argument('--features-dir', type=str, default='./features')
    parser.add_argument('--output-dir', type=str, default='./clustering_results')
    
    args = parser.parse_args()
    
    clusterer = MarketingClusterer(
        features_dir=args.features_dir,
        output_dir=args.output_dir
    )
    
    success = clusterer.run()
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
