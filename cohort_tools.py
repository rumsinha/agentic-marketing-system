#!/usr/bin/env python3
"""Cohort Tools - Google ADK Function Tools for Marketing Intelligence.

Tools that READ from:
1. marketing_playbook.json (segment definitions, actions, KPIs)
2. Snowflake tables (customer_intelligence, cohort_profiles, cluster_profiles)

This module follows Google ADK patterns:
- Pure functions with Google-style docstrings
- ToolContext for accessing session state
- Returns dict with status field
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from google.adk.tools.tool_context import ToolContext

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# ENUMS
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


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class QueryAnalysis:
    """Result of analyzing a user query."""
    intent: QueryIntent
    column_selection: ColumnSelection
    confidence: float
    reasoning: str
    extracted_params: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# KEYWORDS FOR INTENT DETECTION
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
    'churning but', 'high value but', 'at risk', 'behaving strangely', 'strangely'
]

VALUE_KEYWORDS = [
    'high value', 'high-value', 'valuable', 'vip', 'premium',
    'big spender', 'frequent', 'loyal', 'champions', 'top', 'best'
]


# ============================================================
# INTENT DETECTION
# ============================================================

def detect_query_intent(query: str) -> QueryAnalysis:
    """Analyze user query to determine intent and column selection.

    Args:
        query (str): The user's natural language query.

    Returns:
        QueryAnalysis: Analysis with intent, column selection, and reasoning.
    """
    query_lower = query.lower()
    
    strategic_score = sum(1 for kw in STRATEGIC_KEYWORDS if kw in query_lower)
    targeting_score = sum(1 for kw in TARGETING_KEYWORDS if kw in query_lower)
    anomaly_score = sum(1 for kw in ANOMALY_KEYWORDS if kw in query_lower)
    value_score = sum(1 for kw in VALUE_KEYWORDS if kw in query_lower)
    
    params = _extract_query_params(query_lower)
    total_score = strategic_score + targeting_score + anomaly_score + 1
    
    # Anomaly + Value → Use both RFM and HDBSCAN
    if anomaly_score > 0 and value_score > 0:
        return QueryAnalysis(
            intent=QueryIntent.ANOMALY,
            column_selection=ColumnSelection.BOTH,
            confidence=min(0.9, (anomaly_score + value_score) / 4),
            reasoning="Anomaly + value indicators → Use both RFM and HDBSCAN",
            extracted_params=params
        )
    
    # Targeting with category → HDBSCAN
    if targeting_score > 0 and params.get("category"):
        return QueryAnalysis(
            intent=QueryIntent.TARGETING,
            column_selection=ColumnSelection.HDBSCAN_CLUSTER,
            confidence=min(0.9, targeting_score / 3),
            reasoning="Targeting with category → Use HDBSCAN clusters",
            extracted_params=params
        )
    
    # Anomaly detection → Both
    if anomaly_score > 0:
        return QueryAnalysis(
            intent=QueryIntent.ANOMALY,
            column_selection=ColumnSelection.BOTH,
            confidence=anomaly_score / total_score,
            reasoning="Anomaly detection → Use both RFM and HDBSCAN",
            extracted_params=params
        )
    
    # Targeting → HDBSCAN
    if targeting_score > strategic_score:
        return QueryAnalysis(
            intent=QueryIntent.TARGETING,
            column_selection=ColumnSelection.HDBSCAN_CLUSTER,
            confidence=targeting_score / total_score,
            reasoning="Campaign targeting → Use HDBSCAN clusters",
            extracted_params=params
        )
    
    # Default: Strategic → RFM
    return QueryAnalysis(
        intent=QueryIntent.STRATEGIC,
        column_selection=ColumnSelection.RFM_SEGMENT,
        confidence=max(0.5, strategic_score / total_score),
        reasoning="Strategic/overview query → Use RFM segments (100% coverage)",
        extracted_params=params
    )


def _extract_query_params(query: str) -> Dict[str, Any]:
    """Extract structured parameters from query string."""
    params = {}
    
    top_match = re.search(r"top\s*(\d+)", query)
    if top_match:
        params["top_n"] = int(top_match.group(1))
    
    categories = ["tech", "electronics", "fashion", "home", "beauty", "health", "sports", "computers"]
    for cat in categories:
        if cat in query:
            params["category"] = cat
            break
    
    states = ["sp", "rj", "mg", "rs", "pr", "sc", "ba"]
    for state in states:
        if f" {state} " in f" {query} " or query.endswith(f" {state}"):
            params["state"] = state.upper()
            break
    
    return params


# ============================================================
# DATA LOADER
# ============================================================

class DataLoader:
    """Data loader for marketing playbook and Snowflake."""
    
    def __init__(self, playbook_path: str = None, snowflake_config: dict = None, use_snowflake: bool = False):
        self.playbook_path = playbook_path or os.getenv("PLAYBOOK_PATH", "./marketing_playbook.json")
        self.snowflake_config = snowflake_config or {}
        self.use_snowflake = use_snowflake
        self._playbook = None
        self._snowflake_conn = None
    
    @property
    def playbook(self) -> Dict[str, Any]:
        if self._playbook is None:
            self._playbook = self._load_playbook()
        return self._playbook
    
    def _load_playbook(self) -> Dict[str, Any]:
        try:
            with open(self.playbook_path, "r", encoding="utf-8") as f:
                playbook = json.load(f)
            logger.info(f"Loaded playbook from {self.playbook_path}")
            logger.info(f"  Total customers: {playbook.get('total_customers', 'N/A')}")
            logger.info(f"  Segments: {len(playbook.get('segments', {}))}")
            return playbook
        except FileNotFoundError:
            logger.warning(f"Playbook not found at {self.playbook_path}")
            return {"segments": {}, "total_customers": 0}
        except Exception as e:
            logger.error(f"Error loading playbook: {e}")
            return {"segments": {}, "total_customers": 0}
    
    def get_snowflake_connection(self):
        if not self.use_snowflake:
            return None
        if self._snowflake_conn is None:
            try:
                import snowflake.connector
                self._snowflake_conn = snowflake.connector.connect(
                    account=self.snowflake_config.get("account") or os.getenv("SNOWFLAKE_ACCOUNT"),
                    user=self.snowflake_config.get("user") or os.getenv("SNOWFLAKE_USER"),
                    password=self.snowflake_config.get("password") or os.getenv("SNOWFLAKE_PASSWORD"),
                    warehouse=self.snowflake_config.get("warehouse", "COMPUTE_WH"),
                    database=self.snowflake_config.get("database", "MARKETING_INTELLIGENCE"),
                    schema=self.snowflake_config.get("schema", "CUSTOMER_ANALYTICS")
                )
                logger.info("Connected to Snowflake")
            except Exception as e:
                logger.error(f"Snowflake connection failed: {e}")
                return None
        return self._snowflake_conn
    
    def execute_snowflake_query(self, sql: str, params: dict = None) -> List[Dict]:
        conn = self.get_snowflake_connection()
        if not conn:
            return []
        try:
            with conn.cursor() as cursor:
                cursor.execute(sql, params or {})
                columns = [desc[0].lower() for desc in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return []
    
    def get_all_segments(self) -> List[Dict]:
        segments = []
        for name, data in self.playbook.get("segments", {}).items():
            segments.append({
                "rfm_segment": name,
                "customer_count": data.get("customer_count", 0),
                "percentage": data.get("percentage", 0),
                "avg_monetary": data.get("avg_customer_value", 0),
                "priority": data.get("priority", 5),
                "marketing_goal": data.get("marketing_goal", ""),
                "description": data.get("description", ""),
                "budget_allocation": data.get("budget_allocation", ""),
                "recommended_actions": data.get("actions", []),
                "kpis": data.get("kpis", [])
            })
        segments.sort(key=lambda x: -(x.get("priority") or 0))
        return segments
    
    def get_segment_details(self, segment_name: str) -> Optional[Dict]:
        for name, data in self.playbook.get("segments", {}).items():
            if segment_name.lower() in name.lower():
                return {
                    "rfm_segment": name,
                    "customer_count": data.get("customer_count", 0),
                    "percentage": data.get("percentage", 0),
                    "avg_monetary": data.get("avg_customer_value", 0),
                    "priority": int(data.get("priority", 5)),
                    "marketing_goal": data.get("marketing_goal", ""),
                    "description": data.get("description", ""),
                    "budget_allocation": data.get("budget_allocation", ""),
                    "recommended_actions": data.get("actions", []),
                    "kpis": data.get("kpis", [])
                }
        return None
    
    def get_clusters_by_category(self, category: str) -> Dict:
        if not self.use_snowflake:
            return {'error': 'Snowflake required for cluster queries', 'clusters': [], 'customers': []}
        sql_clusters = """SELECT cluster_id, size, primary_cohort, personalization_hint,
                         ROUND(avg_monetary, 2) AS avg_monetary
                         FROM cluster_profiles
                         WHERE LOWER(personalization_hint) LIKE LOWER(%(category_pattern)s)
                         ORDER BY size DESC"""
        clusters = self.execute_snowflake_query(sql_clusters, {'category_pattern': f'%{category}%'})
        return {'clusters': clusters, 'total_clusters': len(clusters)}
    
    def detect_anomalies(self) -> Dict:
        if not self.use_snowflake:
            return {'error': 'Snowflake required for anomaly detection', 'anomalies': []}
        sql = """SELECT customer_unique_id, rfm_segment, hdbscan_cluster, monetary
                 FROM customer_intelligence
                 WHERE rfm_segment IN ('Champions', 'Loyal Customers', 'At Risk')
                 AND hdbscan_cluster = -1"""
        anomalies = self.execute_snowflake_query(sql)
        return {
            'anomaly_type': 'Value-Behavior Mismatch',
            'detection_logic': 'High RFM value + HDBSCAN noise',
            'anomalies': anomalies,
            'total_anomalies': len(anomalies)
        }
    
    def close(self):
        if self._snowflake_conn:
            self._snowflake_conn.close()


# ============================================================
# GLOBAL DATA LOADER
# ============================================================

_data_loader: Optional[DataLoader] = None


def get_data_loader() -> DataLoader:
    """Get or create the global DataLoader instance."""
    global _data_loader
    if _data_loader is None:
        _data_loader = DataLoader()
    return _data_loader


def initialize_data_loader(playbook_path: str = None, snowflake_config: dict = None, use_snowflake: bool = False):
    """Initialize the global data loader with custom configuration.

    Args:
        playbook_path (str): Path to marketing_playbook.json file.
        snowflake_config (dict): Snowflake connection configuration.
        use_snowflake (bool): Whether to use Snowflake for queries.
    """
    global _data_loader
    _data_loader = DataLoader(playbook_path, snowflake_config, use_snowflake)


# ============================================================
# ENTITY EXTRACTION
# ============================================================

COHORT_KEYWORDS = [
    "champions", "loyal", "at risk", "about to sleep", "potential loyalist",
    "recent", "need attention", "promising", "hibernating", "lost", "price sensitive"
]


def extract_segments_from_text(text: str) -> List[str]:
    """Extract segment names from text.

    Args:
        text (str): Text to search for segment names.

    Returns:
        List[str]: List of found segment names (title case).
    """
    text_lower = text.lower()
    clean_text = re.sub(r'[^\w\s]', ' ', text_lower)
    tokens = set(clean_text.split())
    
    found = []
    for kw in COHORT_KEYWORDS:
        if " " in kw:
            if kw in text_lower:
                found.append(kw.title())
        elif kw in tokens:
            found.append(kw.title())
    return found


# ============================================================
# ADK FUNCTION TOOLS
# These functions are used by the ADK Agent.
# They receive tool_context for accessing session state.
# ============================================================

def smart_customer_query(query: str, tool_context: ToolContext = None) -> dict:
    """Query customer data with intelligent routing to RFM segments or HDBSCAN clusters.

    Automatically analyzes the query intent and routes to the appropriate data source:
    - Strategic questions use RFM segments (100% customer coverage)
    - Targeting questions use HDBSCAN clusters (behavioral patterns)
    - Anomaly detection uses both sources

    Args:
        query (str): Natural language query about customer segments.

    Returns:
        dict: Query results with segments, analysis metadata, and recommendations.
    """
    # --- ADD STATE LOGIC ---
    if tool_context:
         # Example: Read a user preference (if it existed)
         user_role = tool_context.state.get("user_role", "general")
    # -----------------------

    data_loader = get_data_loader()
    analysis = detect_query_intent(query)
    
    logger.info(f"Smart Query: {query}")
    logger.info(f"  Intent: {analysis.intent.value}")
    logger.info(f"  Column: {analysis.column_selection.value}")
    
    # Route to appropriate data source based on intent
    if analysis.column_selection == ColumnSelection.RFM_SEGMENT:
        top_n = analysis.extracted_params.get("top_n", 5)
        segments = data_loader.get_all_segments()[:top_n]
        
        return {
            'type': 'segment_overview',
            'data_source': 'rfm_segment',
            'reasoning': analysis.reasoning,
            'segments': segments,
            'total_segments': len(segments)
        }
    
    elif analysis.column_selection == ColumnSelection.HDBSCAN_CLUSTER:
        category = analysis.extracted_params.get("category", "electronics")
        result = data_loader.get_clusters_by_category(category)
        result['type'] = 'behavioral_targeting'
        result['data_source'] = 'hdbscan_cluster'
        result['reasoning'] = analysis.reasoning
        return result
    
    else:  # BOTH - Anomaly detection
        result = data_loader.detect_anomalies()
        result['type'] = 'anomaly_detection'
        result['data_source'] = 'both_rfm_and_hdbscan'
        result['reasoning'] = analysis.reasoning
        result['recommended_actions'] = [
            'Review purchase history for pattern changes',
            'Personal outreach to understand behavior',
            'Consider for VIP intervention program'
        ]
        return result


def get_segment_details(segment_name: str,tool_context: ToolContext = None) -> dict:
    """Get detailed information about a specific RFM customer segment.

    Retrieves comprehensive segment data including customer count, average value,
    marketing goals, recommended actions, and KPIs from the marketing playbook.

    Args:
        segment_name (str): Name of the segment to retrieve (e.g., "Champions", "At Risk").

    Returns:
        dict: Segment details with customer_count, avg_monetary, marketing_goal,
              recommended_actions, and kpis. Returns error if segment not found.
    """

    # --- ADD STATE LOGIC ---
    if tool_context:
        # Save this segment as the "active topic" in session state
        tool_context.state["last_segment_discussed"] = segment_name
        logger.info(f"State Updated: last_segment_discussed = {segment_name}")
    # -----------------------

    logger.info(f"Getting segment details for: {segment_name}")
    data_loader = get_data_loader()
    segment = data_loader.get_segment_details(segment_name)
    
    if not segment:
        return {'error': f"Segment '{segment_name}' not found"}
    
    return {
        'type': 'segment_details',
        'data_source': 'rfm_segment',
        'segment': segment
    }


def find_behavioral_cluster(category: str, tool_context: ToolContext = None) -> dict:
    """Find customers by product category using HDBSCAN behavioral clusters.

    Searches for customer clusters that show affinity for a specific product
    category. Requires Snowflake connection for behavioral data.

    Args:
        category (str): Product category to filter by (e.g., "electronics", "fashion").

    Returns:
        dict: Matching clusters with customer counts and behavioral hints.
    """
    if not category or not category.strip():
        return {'error': "Missing parameter: 'category'. Please specify a product category."}
    
    data_loader = get_data_loader()
    result = data_loader.get_clusters_by_category(category)
    result['type'] = 'behavioral_targeting'
    result['data_source'] = 'hdbscan_cluster'
    return result


def detect_customer_anomalies(tool_context: ToolContext = None) -> dict:
    """Detect high-value customers exhibiting unusual behavioral patterns.

    Identifies customers who have high RFM value (Champions, Loyal, At Risk)
    but appear as noise (-1) in HDBSCAN clustering, indicating potential
    issues requiring attention.

    Returns:
        dict: List of anomalous customers with RFM segment, cluster, and monetary value.
    """
    data_loader = get_data_loader()
    result = data_loader.detect_anomalies()
    result['type'] = 'anomaly_detection'
    result['data_source'] = 'both_rfm_and_hdbscan'
    result['recommended_actions'] = [
        'Review purchase history for pattern changes',
        'Personal outreach to understand behavior',
        'Consider for VIP intervention program'
    ]
    return result

from typing import List  # Ensure this is imported at the top

def compare_segments(segment_names: List[str], tool_context: ToolContext = None) -> dict:
    """Compare multiple RFM segments side by side and generate insights.

    Analyzes the differences between segments including customer count,
    average monetary value, priority, and generates comparison insights.

    Args:
        segment_names (list): List of segment names to compare (e.g., ["Champions", "At Risk"]).

    Returns:
        dict: Comparison data with segment details and generated insights.
    """

    # --- ADD STATE LOGIC ---
    if tool_context and len(segment_names) == 1:
        # Smart Feature: If user only says "Compare with Lost", fetch the OTHER one from history
        last_segment = tool_context.state.get("last_segment_discussed")
        if last_segment and last_segment not in segment_names:
            segment_names.insert(0, last_segment)
            logger.info(f"State Context: Auto-added '{last_segment}' to comparison")
    # -----------------------

    if not segment_names:
        return {'error': 'No segments provided for comparison'}
    
    # Handle string input
    if isinstance(segment_names, str):
        segment_names = [segment_names]
    
    data_loader = get_data_loader()
    segments = []
    for name in segment_names:
        seg = data_loader.get_segment_details(name)
        if seg:
            segments.append(seg)
    
    # Generate insights
    insights = []
    if len(segments) >= 2:
        s1, s2 = segments[0], segments[1]
        if s1['avg_monetary'] > s2['avg_monetary'] and s2['avg_monetary'] > 0:
            ratio = s1['avg_monetary'] / s2['avg_monetary']
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


# ============================================================
# CLI
# ============================================================

def main():
    import argparse
    from dotenv import load_dotenv
    load_dotenv()
    
    parser = argparse.ArgumentParser(description='Cohort Tools')
    parser.add_argument('--playbook', type=str, default='./marketing_playbook.json')
    parser.add_argument('--query', type=str, help='Query to test')
    parser.add_argument('--test', action='store_true', help='Run tests')
    
    args = parser.parse_args()
    initialize_data_loader(args.playbook)
    
    if args.test:
        print("=" * 60)
        print("TESTING COHORT TOOLS")
        print("=" * 60)
        
        print("\n1. Strategic Query:")
        result = smart_customer_query("Show me top 3 segments")
        print(f"   Segments: {len(result.get('segments', []))}")
        
        print("\n2. Segment Details:")
        result = get_segment_details("Champions")
        if 'segment' in result:
            print(f"   {result['segment']['rfm_segment']}: {result['segment']['customer_count']} customers")
        
        print("\n3. Compare Segments:")
        result = compare_segments(["Champions", "Lost"])
        print(f"   Insights: {result.get('insights', [])}")
        
        print("\n" + "=" * 60)
    elif args.query:
        print(json.dumps(smart_customer_query(args.query), indent=2, default=str))
    else:
        print("Usage: python cohort_tools.py --test")


if __name__ == "__main__":
    main()