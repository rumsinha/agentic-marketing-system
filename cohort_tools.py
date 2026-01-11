#!/usr/bin/env python3
"""
COHORT TOOLS 
============================
Tools that READ from:
1. marketing_playbook.json (segment definitions, actions, KPIs)
2. Snowflake tables (customer_intelligence, cohort_profiles, cluster_profiles)

NO hardcoded segment definitions - all data comes from the ML pipeline outputs.

Data Sources:
- marketing_playbook.json: Segment definitions, actions, KPIs, budget allocation
- Snowflake.customer_intelligence: 95K customers with RFM + HDBSCAN
- Snowflake.rfm_segment_definitions: 11 segment profiles
- Snowflake.cluster_profiles: 89 HDBSCAN cluster profiles

"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# QUERY INTENT CLASSIFICATION
# ============================================================

class QueryIntent(Enum):
    """Classification of user query intent."""
    STRATEGIC = "strategic"
    TARGETING = "targeting"
    ANOMALY = "anomaly"
    COMPARISON = "comparison"
    SEARCH = "search"
    ACTIONS = "actions"


class ColumnSelection(Enum):
    """Which column(s) to use for the query."""
    RFM_SEGMENT = "rfm_segment"
    HDBSCAN_CLUSTER = "hdbscan_cluster"
    BOTH = "both"


@dataclass
class QueryAnalysis:
    """Result of analyzing a user query."""
    intent: QueryIntent
    column_selection: ColumnSelection
    confidence: float
    reasoning: str
    extracted_params: Dict[str, Any]


# ============================================================
# INTENT DETECTION (Keyword-based)
# ============================================================

STRATEGIC_KEYWORDS = [
    'best', 'top', 'overview', 'summary', 'report', 'dashboard',
    'performing', 'valuable', 'important', 'priority', 'show me',
    'how many', 'total', 'all segments', 'cohorts', 'breakdown',
    'revenue', 'health', 'status', 'kpis', 'metrics'
]

TARGETING_KEYWORDS = [
    'campaign', 'target', 'personalize', 'email', 'marketing',
    'buy', 'purchase', 'category', 'product', 'tech', 'electronics',
    'fashion', 'behavior', 'pattern', 'similar', 'like', 'prefer',
    'interest', 'segment for', 'list for', 'audience', 'cluster'
]

ANOMALY_KEYWORDS = [
    'weird', 'unusual', 'anomaly', 'outlier', 'strange', 'different',
    'acting', 'behavior change', 'unexpected', 'noise', 'exception',
    'not fitting', 'edge case', 'investigate', 'why', 'dropping',
    'churning but', 'high value but', 'at risk'
]

VALUE_KEYWORDS = [
    'high value', 'high-value', 'valuable', 'vip', 'premium',
    'big spender', 'frequent', 'loyal', 'champions'
]


def detect_query_intent(query: str) -> QueryAnalysis:
    """Analyze user query to determine intent and column selection."""
    query_lower = query.lower()
    
    # Count keyword matches
    strategic_score = sum(1 for kw in STRATEGIC_KEYWORDS if kw in query_lower)
    targeting_score = sum(1 for kw in TARGETING_KEYWORDS if kw in query_lower)
    anomaly_score = sum(1 for kw in ANOMALY_KEYWORDS if kw in query_lower)
    value_score = sum(1 for kw in VALUE_KEYWORDS if kw in query_lower)
    
    # Extract parameters
    params = _extract_query_params(query_lower)
    
    total_score = strategic_score + targeting_score + anomaly_score + 1 # The `+ 1` prevents division by zero.
    
    # Decision logic
    # Important customers are behaving strangely
    # The confidence score represents how certain the system is about its intent classification - essentially "how sure are we that we understood the user correctly?"
    if anomaly_score > 0 and value_score > 0:
        return QueryAnalysis(
            intent=QueryIntent.ANOMALY,
            column_selection=ColumnSelection.BOTH,
            confidence=min(0.9, (anomaly_score + value_score) / 4), # Cap at 90% (never be 100% sure with keyword matching)
            reasoning="Anomaly + value indicators → Use both RFM and HDBSCAN",
            extracted_params=params
        )
    
    if targeting_score > 0 and params.get('category'):
        return QueryAnalysis(
            intent=QueryIntent.TARGETING,
            column_selection=ColumnSelection.HDBSCAN_CLUSTER,
            confidence=min(0.9, targeting_score / 3), # 3 keywords = confident enough
            reasoning="Targeting with category → Use HDBSCAN clusters",
            extracted_params=params
        )
    
    if anomaly_score > 0:
        return QueryAnalysis(
            intent=QueryIntent.ANOMALY,
            column_selection=ColumnSelection.BOTH,
            confidence=anomaly_score / total_score,
            reasoning="Anomaly detection → Use both RFM and HDBSCAN",
            extracted_params=params
        )
    
    if targeting_score > strategic_score:
        return QueryAnalysis(
            intent=QueryIntent.TARGETING,
            column_selection=ColumnSelection.HDBSCAN_CLUSTER,
            confidence=targeting_score / total_score,
            reasoning="Campaign targeting → Use HDBSCAN clusters",
            extracted_params=params
        )
    
    # Default: Strategic
    return QueryAnalysis(
        intent=QueryIntent.STRATEGIC,
        column_selection=ColumnSelection.RFM_SEGMENT,
        confidence=max(0.5, strategic_score / total_score),
        reasoning="Strategic/overview query → Use RFM segments (100% coverage)",
        extracted_params=params
    )


def _extract_query_params(query: str) -> Dict[str, Any]:
    """Extract parameters from query."""
    import re
    params = {}
    
    # Extract top_n
    top_match = re.search(r'top\s*(\d+)', query)
    if top_match:
        params['top_n'] = int(top_match.group(1))
    
    # Extract categories
    categories = ['tech', 'electronics', 'fashion', 'home', 'beauty', 'health', 'sports', 'computers']
    for cat in categories:
        if cat in query:
            params['category'] = cat
            break
    
    # Extract states
    states = ['sp', 'rj', 'mg', 'rs', 'pr', 'sc', 'ba']
    for state in states:
        if f' {state} ' in f' {query} ' or query.endswith(f' {state}'):
            params['state'] = state.upper()
            break
    
    return params


# ============================================================
# DATA LOADER - Reads from marketing_playbook.json and Snowflake
# ============================================================

class DataLoader:
    """
    Loads data from marketing_playbook.json and Snowflake.
    NO hardcoded data - everything comes from ML pipeline outputs.
    """
    
    def __init__(
        self,
        playbook_path: str = None,
        snowflake_config: Dict = None,
        use_snowflake: bool = False
    ):
        self.playbook_path = playbook_path or os.getenv("PLAYBOOK_PATH", "./marketing_playbook.json")
        self.snowflake_config = snowflake_config
        self.use_snowflake = use_snowflake
        
        # Cached data
        self._playbook = None
        self._snowflake_conn = None
    
    @property
    def playbook(self) -> Dict:
        """Load and cache marketing playbook."""
        if self._playbook is None:
            self._playbook = self._load_playbook()
        return self._playbook
    
    def _load_playbook(self) -> Dict:
        """Load marketing_playbook.json."""
        try:
            with open(self.playbook_path, 'r') as f:
                playbook = json.load(f)
            logger.info(f"Loaded playbook from {self.playbook_path}")
            logger.info(f"  Total customers: {playbook.get('total_customers', 'N/A')}")
            logger.info(f"  Segments: {len(playbook.get('segments', {}))}")
            return playbook
        except FileNotFoundError:
            logger.warning(f"Playbook not found at {self.playbook_path}, using empty playbook")
            return {"segments": {}, "total_customers": 0}
        except Exception as e:
            logger.error(f"Error loading playbook: {e}")
            return {"segments": {}, "total_customers": 0}
    
    def get_snowflake_connection(self):
        """Get Snowflake connection."""
        if not self.use_snowflake:
            return None
        
        if self._snowflake_conn is None:
            try:
                import snowflake.connector
                self._snowflake_conn = snowflake.connector.connect(
                    account=self.snowflake_config.get('account') or os.getenv('SNOWFLAKE_ACCOUNT'),
                    user=self.snowflake_config.get('user') or os.getenv('SNOWFLAKE_USER'),
                    password=self.snowflake_config.get('password') or os.getenv('SNOWFLAKE_PASSWORD'),
                    warehouse=self.snowflake_config.get('warehouse', 'COMPUTE_WH'),
                    database=self.snowflake_config.get('database', 'MARKETING_INTELLIGENCE'),
                    schema=self.snowflake_config.get('schema', 'CUSTOMER_ANALYTICS')
                )
                logger.info("Connected to Snowflake")
            except Exception as e:
                logger.error(f"Snowflake connection failed: {e}")
                return None
        
        return self._snowflake_conn
    
    def execute_snowflake_query(self, sql: str, params: Dict = None) -> List[Dict]:
        """Execute Snowflake query and return results as list of dicts."""
        conn = self.get_snowflake_connection()
        if not conn:
            return []
        
        try:
            # Use context manager 'with' to auto-close cursor
            with conn.cursor() as cursor:
                cursor.execute(sql, params or {})
                columns = [desc[0].lower() for desc in cursor.description]
                rows = cursor.fetchall()
                return [dict(zip(columns, row)) for row in rows]
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return []
    
    # ========================================
    # Playbook-based queries (no Snowflake)
    # ========================================
    
    def get_all_segments(self) -> List[Dict]:
        """Get all segments from playbook."""
        segments = []
        for name, data in self.playbook.get('segments', {}).items():
            segments.append({
                'rfm_segment': name,
                'customer_count': data.get('customer_count', 0),
                'percentage': data.get('percentage', 0),
                'avg_monetary': data.get('avg_customer_value', 0),
                'priority': data.get('priority', 5),
                'marketing_goal': data.get('marketing_goal', ''),
                'description': data.get('description', ''),
                'budget_allocation': data.get('budget_allocation', ''),
                'recommended_actions': data.get('actions', []),
                'kpis': data.get('kpis', [])
            })
        
        # Sort by priority descending
        segments.sort(key=lambda x: -x['priority'])
        return segments
    
    def get_segment_details(self, segment_name: str) -> Optional[Dict]:
        """Get segment details by name from playbook."""
        segments = self.playbook.get('segments', {})
        
        # Case-insensitive search
        for name, data in segments.items():
            if segment_name.lower() in name.lower():
                return {
                    'rfm_segment': name,
                    'customer_count': data.get('customer_count', 0),
                    'percentage': data.get('percentage', 0),
                    'avg_monetary': data.get('avg_customer_value', 0),
                    'priority': int(data.get('priority', 5)),
                    'marketing_goal': data.get('marketing_goal', ''),
                    'description': data.get('description', ''),
                    'budget_allocation': data.get('budget_allocation', ''),
                    'recommended_actions': data.get('actions', []),
                    'kpis': data.get('kpis', [])
                }
        
        return None
    
    def get_segment_names(self) -> List[str]:
        """Get list of all segment names."""
        return list(self.playbook.get('segments', {}).keys())
    
    # ========================================
    # Snowflake-based queries (for customer-level data)
    # ========================================
    
    def get_segment_summary_from_snowflake(self, top_n: int = 10) -> List[Dict]:
        """Get segment summary from Snowflake."""
        if not self.use_snowflake:
            # Fallback to playbook
            return self.get_all_segments()[:top_n]
        
        sql = """
                SELECT 
            cohort_name AS rfm_segment,
            customer_count,
            ROUND(avg_monetary, 2) AS avg_monetary,
            priority,
            marketing_goal,
            recommended_actions,
            kpis
        FROM cohort_profiles
        ORDER BY priority DESC
        """
        
        results = self.execute_snowflake_query(sql, {'top_n': top_n})
        
        # Enrich with playbook data (actions, KPIs)
        for row in results:
            playbook_data = self.get_segment_details(row['rfm_segment'])
            if playbook_data:
                row['marketing_goal'] = playbook_data.get('marketing_goal', '')
                row['recommended_actions'] = playbook_data.get('recommended_actions', [])
                row['kpis'] = playbook_data.get('kpis', [])
        
        return results
    
    def get_clusters_by_category(self, category: str) -> Dict:
        """Get HDBSCAN clusters by product category from Snowflake."""
        if not self.use_snowflake:
            return {'error': 'Snowflake required for cluster queries', 'clusters': [], 'customers': []}
        
        # Get cluster profiles
        sql_clusters = """
        SELECT 
            cluster_id, 
            size, 
            primary_cohort, 
            personalization_hint, 
            ROUND(avg_monetary, 2) AS avg_monetary,
            ROUND(avg_rfm_total, 2) AS avg_rfm_score
        FROM cluster_profiles
        WHERE LOWER(personalization_hint) LIKE LOWER(%(category_pattern)s)
        OR LOWER(primary_cohort) LIKE LOWER(%(cohort_pattern)s)
        ORDER BY size DESC
        """
        
        # Parameters matching the SQL placeholders
        clusters = self.execute_snowflake_query(sql_clusters, {'category_pattern': f'%{category}%', 'cohort_pattern': f'%{category}%'})

        if not clusters:
            return {'clusters': [], 'customers': [], 'total_customers': 0}
        
        # Get customers in those clusters
        cluster_ids = [c['cluster_id'] for c in clusters]
        placeholders = ','.join(str(c) for c in cluster_ids)
        
        sql_customers = f"""
        SELECT customer_unique_id, rfm_segment, hdbscan_cluster, monetary
        FROM customer_intelligence
        WHERE hdbscan_cluster IN ({placeholders})
        ORDER BY monetary DESC
        LIMIT 100
        """
        
        customers = self.execute_snowflake_query(sql_customers)
        
        return {
            'target_category': category,
            'clusters': clusters,
            'customers': customers,
            'total_customers': len(customers)
        }
    
    def detect_anomalies(self) -> Dict:
        """Detect high-value customers with unusual behavior."""
        if not self.use_snowflake:
            return {'error': 'Snowflake required for anomaly detection', 'anomalies': []}
        
        sql = """
                SELECT 
            customer_unique_id,
            rfm_segment,
            hdbscan_cluster,
            ROUND(monetary, 2) AS monetary,
            recency,
            frequency,
            priority
        FROM customer_intelligence
        WHERE rfm_segment IN ('Champions', 'Loyal Customers', 'At Risk')
        AND hdbscan_cluster = -1
        """
        
        anomalies = self.execute_snowflake_query(sql)
        
        return {
            'anomaly_type': 'Value-Behavior Mismatch',
            'detection_logic': 'High RFM value + HDBSCAN noise/low probability',
            'anomalies': anomalies,
            'total_anomalies': len(anomalies)
        }
    
    def search_customers(
        self,
        rfm_segment: str = None,
        min_monetary: float = None,
        state: str = None,
        limit: int = 100
    ) -> List[Dict]:
        """Search customers with filters."""
        if not self.use_snowflake:
            return []
        
        conditions = ["1=1"]
        params = {'limit': limit}
        
        if rfm_segment:
            conditions.append("LOWER(rfm_segment) LIKE LOWER(%(rfm_segment)s)")
            params['rfm_segment'] = f'%{rfm_segment}%'
        
        if min_monetary:
            conditions.append("monetary >= %(min_monetary)s")
            params['min_monetary'] = min_monetary
        
        if state:
            conditions.append("UPPER(customer_state) = UPPER(%(state)s)")
            params['state'] = state
        
        sql = f"""
        SELECT customer_unique_id, rfm_segment, hdbscan_cluster, monetary, recency, frequency
        FROM customer_intelligence
        WHERE {' AND '.join(conditions)}
        ORDER BY monetary DESC
        LIMIT %(limit)s
        """
        
        return self.execute_snowflake_query(sql, params)
    
    def close(self):
        """Close connections."""
        if self._snowflake_conn:
            self._snowflake_conn.close()


# ============================================================
# COHORT TOOLS - Uses DataLoader (no hardcoded data)
# ============================================================

class CohortTools:
    """
    Tools for querying cohort data.
    Reads from marketing_playbook.json and Snowflake.
    """
    
    def __init__(
        self,
        playbook_path: str = None,
        snowflake_config: Dict = None,
        use_snowflake: bool = False
    ):
        self.data_loader = DataLoader(
            playbook_path=playbook_path,
            snowflake_config=snowflake_config,
            use_snowflake=use_snowflake
        )
    
    def smart_query(self, query: str) -> Dict[str, Any]:
        """
        Intelligently route query to appropriate data source.
        """
        analysis = detect_query_intent(query)
        
        logger.info(f"Smart Query: {query}")
        logger.info(f"  Intent: {analysis.intent.value}")
        logger.info(f"  Column: {analysis.column_selection.value}")
        
        if analysis.column_selection == ColumnSelection.RFM_SEGMENT:
            result = self._handle_rfm_query(query, analysis)
        elif analysis.column_selection == ColumnSelection.HDBSCAN_CLUSTER:
            result = self._handle_cluster_query(query, analysis)
        else:
            result = self._handle_hybrid_query(query, analysis)
        
        result['query_analysis'] = {
            'intent': analysis.intent.value,
            'column_selection': analysis.column_selection.value,
            'confidence': analysis.confidence,
            'reasoning': analysis.reasoning
        }
        
        return result
    
    def _handle_rfm_query(self, query: str, analysis: QueryAnalysis) -> Dict:
        """Handle strategic queries using RFM segments."""
        params = analysis.extracted_params
        top_n = params.get('top_n', 5)
        
        if self.data_loader.use_snowflake:
            segments = self.data_loader.get_segment_summary_from_snowflake(top_n)
        else:
            segments = self.data_loader.get_all_segments()[:top_n]
        
        return {
            'type': 'segment_overview',
            'data_source': 'rfm_segment',
            'coverage': '100%',
            'segments': segments,
            'total_segments': len(segments),
            'note': 'Using RFM segments for complete customer coverage'
        }
    
    def _handle_cluster_query(self, query: str, analysis: QueryAnalysis) -> Dict:
        """Handle targeting queries using HDBSCAN clusters."""
        params = analysis.extracted_params
        category = params.get('category', 'electronics')
        
        result = self.data_loader.get_clusters_by_category(category)
        result['type'] = 'behavioral_targeting'
        result['data_source'] = 'hdbscan_cluster'
        
        return result
    
    def _handle_hybrid_query(self, query: str, analysis: QueryAnalysis) -> Dict:
        """Handle anomaly detection using both RFM and HDBSCAN."""
        result = self.data_loader.detect_anomalies()
        result['type'] = 'anomaly_detection'
        result['data_source'] = 'both_rfm_and_hdbscan'
        result['recommended_actions'] = [
            'Review purchase history for pattern changes',
            'Personal outreach to understand behavior',
            'Consider for VIP intervention program'
        ]
        
        return result
    
    def get_segment_details(self, segment_name: str) -> Dict:
        """Get detailed information about a specific segment."""
        segment = self.data_loader.get_segment_details(segment_name)
        
        if not segment:
            return {'error': f"Segment '{segment_name}' not found"}
        
        return {
            'type': 'segment_details',
            'data_source': 'rfm_segment',
            'segment': segment
        }
    
    def compare_segments(self, segment_names: List[str]) -> Dict:
        """Compare multiple segments."""
        segments = []
        for name in segment_names:
            seg = self.data_loader.get_segment_details(name)
            if seg:
                segments.append(seg)
        
        # Generate insights
        insights = []
        if len(segments) >= 2:
            s1, s2 = segments[0], segments[1]
            if s1['avg_monetary'] > s2['avg_monetary']:
                ratio = s1['avg_monetary'] / s2['avg_monetary'] if s2['avg_monetary'] > 0 else 0
                insights.append(f"{s1['rfm_segment']} has {ratio:.1f}x higher average spend")
            if s1['priority'] != s2['priority']:
                higher = s1 if s1['priority'] > s2['priority'] else s2
                insights.append(f"{higher['rfm_segment']} has higher marketing priority")
        
        return {
            'type': 'segment_comparison',
            'data_source': 'rfm_segment',
            'segments': segments,
            'insights': insights
        }
    
    def explain_column_selection(self, query: str) -> Dict:
        """Explain why a particular column was selected."""
        analysis = detect_query_intent(query)
        
        explanations = {
            ColumnSelection.RFM_SEGMENT: {
                'column': 'rfm_segment',
                'why': '100% coverage, business-meaningful segments, pre-defined actions',
                'best_for': 'Strategic reporting, KPIs, dashboards'
            },
            ColumnSelection.HDBSCAN_CLUSTER: {
                'column': 'hdbscan_cluster',
                'why': 'Behavioral patterns, category affinity, personalization',
                'best_for': 'Campaign targeting, personalization'
            },
            ColumnSelection.BOTH: {
                'column': 'both',
                'why': 'Detect value-behavior mismatches, find outliers',
                'best_for': 'Anomaly detection, VIP monitoring'
            }
        }
        
        return {
            'query': query,
            'selected': analysis.column_selection.value,
            'intent': analysis.intent.value,
            'reasoning': analysis.reasoning,
            'explanation': explanations[analysis.column_selection]
        }


# ============================================================
# CLI FOR TESTING
# ============================================================

def main():
    import argparse
    import os
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(description='Cohort Tools')
    parser.add_argument('--playbook', type=str, default='./marketing_playbook.json')
    parser.add_argument('--query', type=str, help='Query to test')
    parser.add_argument('--use-snowflake', action='store_true')
    parser.add_argument('--test', action='store_true', help='Run tests')
    
    args = parser.parse_args()

    # Create the config dictionary
    sf_config = {
        "account": os.getenv("SNOWFLAKE_ACCOUNT"),
        "user": os.getenv("SNOWFLAKE_USER"),
        "password": os.getenv("SNOWFLAKE_PASSWORD"),
        "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"),
        "database": os.getenv("SNOWFLAKE_DATABASE", "MARKETING_INTELLIGENCE"),
        "schema": os.getenv("SNOWFLAKE_SCHEMA", "CUSTOMER_ANALYTICS")
    }
    
    tools = CohortTools(
        playbook_path=args.playbook,
        snowflake_config=sf_config,
        use_snowflake=args.use_snowflake
    )
    
    if args.test:
        print("=" * 60)
        print("TESTING COHORT TOOLS")
        print("=" * 60)
        
        # Test 1: Strategic query
        print("\n1. Strategic Query:")
        result = tools.smart_query("Show me our best performing customers")
        print(f"   Intent: {result['query_analysis']['intent']}")
        print(f"   Column: {result['query_analysis']['column_selection']}")
        print(f"   Segments: {len(result.get('segments', []))}")
        
        # Test 2: Get segment details
        print("\n2. Segment Details:")
        result = tools.get_segment_details("Champions")
        if 'segment' in result:
            seg = result['segment']
            print(f"   {seg['rfm_segment']}: {seg['customer_count']} customers")
            print(f"   Actions: {seg['recommended_actions'][:2]}")
        
        # Test 3: Compare segments
        print("\n3. Compare Segments:")
        result = tools.compare_segments(["Champions", "Lost"])
        print(f"   Segments compared: {len(result['segments'])}")
        print(f"   Insights: {result['insights']}")
        
        print("\n" + "=" * 60)
        print("TESTS COMPLETE")
        print("=" * 60)
    
    elif args.query:
        result = tools.smart_query(args.query)
        print(json.dumps(result, indent=2, default=str))
    
    else:
        print("Usage: python cohort_tools.py --test")
        print("       python cohort_tools.py --query 'Show me top 3 segments'")


if __name__ == "__main__":
    main()