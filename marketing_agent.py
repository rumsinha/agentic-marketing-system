#!/usr/bin/env python3
"""
MARKETING INTELLIGENCE AGENT - COMPLETE
========================================
Google ADK + Gemini 2.5 Flash + Memory + Context Engineering

Data Sources (NO hardcoded data):
- marketing_playbook.json: Segment definitions, actions, KPIs
- Snowflake: customer_intelligence, cluster_profiles, cohort_profiles

Features:
- Intelligent Query Routing (RFM vs HDBSCAN vs Both)
- 5 MCP Tools
- Persistent Memory (SQLite)
- Context Engineering (entity extraction, summarization)
- Cross-session memory

Usage:
    python adk_marketing_agent.py
    python adk_marketing_agent.py --query "Show me top 3 segments"
    python adk_marketing_agent.py --conversation-id abc123
    python adk_marketing_agent.py --demo

"""
import re
import os
import sys
import json
import uuid
import sqlite3
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
logger = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv()

# Google ADK and Gemini imports
try:
    from google import genai
    from google.genai import types
    ADK_AVAILABLE = True
except ImportError:
    ADK_AVAILABLE = False
    print("Warning: google-genai not installed. Run: pip install google-genai")

sys.path.insert(0, str(Path(__file__).parent))

from cohort_tools import CohortTools, detect_query_intent, QueryIntent, ColumnSelection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class AgentConfig:
    """Configuration for the Marketing Agent."""
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    model_name: str = "gemini-2.5-flash"
    temperature: float = 0.1
    max_output_tokens: int = 4096
    
    playbook_path: str = field(default_factory=lambda: os.getenv("PLAYBOOK_PATH", "./marketing_playbook.json"))
    use_snowflake: bool = False
    
    snowflake_account: str = field(default_factory=lambda: os.getenv("SNOWFLAKE_ACCOUNT", ""))
    snowflake_user: str = field(default_factory=lambda: os.getenv("SNOWFLAKE_USER", ""))
    snowflake_password: str = field(default_factory=lambda: os.getenv("SNOWFLAKE_PASSWORD", ""))
    
    memory_db_path: str = "./checkpoints/agent_memory.db"
    max_messages: int = 20


# ============================================================
# MEMORY SYSTEM
# ============================================================

class MemoryManager:
    """SQLite-based persistent memory."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS conversations (
                conversation_id TEXT PRIMARY KEY, user_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
            conn.execute("""CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY, conversation_id TEXT, role TEXT, content TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
            conn.execute("""CREATE TABLE IF NOT EXISTS context (
                conversation_id TEXT PRIMARY KEY, segments_discussed TEXT,
                facts TEXT, last_intent TEXT, last_data_source TEXT)""")
            conn.execute("""CREATE TABLE IF NOT EXISTS long_term_memory (
                user_id TEXT, conversation_id TEXT, summary TEXT, key_facts TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, PRIMARY KEY (user_id, conversation_id))""")
            conn.commit()
    
    def create_conversation(self, user_id: str = "default") -> str:
        conv_id = str(uuid.uuid4())[:8]
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("INSERT INTO conversations (conversation_id, user_id) VALUES (?, ?)", (conv_id, user_id))
            conn.execute("INSERT INTO context (conversation_id, segments_discussed, facts) VALUES (?, '[]', '[]')", (conv_id,))
            conn.commit()
        return conv_id
    
    def add_message(self, conv_id: str, role: str, content: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)", (conv_id, role, content))
            conn.commit()
    
    def get_messages(self, conv_id: str, limit: int = 20) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY timestamp DESC LIMIT ?", (conv_id, limit))
            return list(reversed([{"role": r[0], "content": r[1]} for r in cursor.fetchall()]))
    
    def get_context(self, conv_id: str) -> Dict:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT segments_discussed, facts, last_intent, last_data_source FROM context WHERE conversation_id = ?", (conv_id,))
            row = cursor.fetchone()
            if row:
                return {"segments_discussed": json.loads(row[0] or "[]"), "facts": json.loads(row[1] or "[]"),
                        "last_intent": row[2], "last_data_source": row[3]}
        return {"segments_discussed": [], "facts": []}
    
    def update_context(self, conv_id: str, **kwargs):
        current = self.get_context(conv_id)
        for key, value in kwargs.items():
            if key in ["segments_discussed", "facts"]:
                if isinstance(value, list):
                    for v in value:
                        if v and v not in current[key]: current[key].append(v)
                elif value and value not in current[key]: current[key].append(value)
            else: current[key] = value
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE context SET segments_discussed=?, facts=?, last_intent=?, last_data_source=? WHERE conversation_id=?",
                        (json.dumps(current["segments_discussed"]), json.dumps(current["facts"]),
                         current.get("last_intent"), current.get("last_data_source"), conv_id))
            conn.commit()
    
    def save_long_term(self, user_id: str, conv_id: str, summary: str, facts: List[str]):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("INSERT OR REPLACE INTO long_term_memory VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)",
                        (user_id, conv_id, summary, json.dumps(facts)))
            conn.commit()
    
    def get_user_memories(self, user_id: str, limit: int = 5) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT conversation_id, summary, key_facts FROM long_term_memory WHERE user_id=? ORDER BY updated_at DESC LIMIT ?", (user_id, limit))
            return [{"conversation_id": r[0], "summary": r[1], "key_facts": json.loads(r[2] or "[]")} for r in cursor.fetchall()]
    
    def exists(self, conv_id: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute("SELECT 1 FROM conversations WHERE conversation_id=?", (conv_id,)).fetchone() is not None


# ============================================================
# CONTEXT ENGINEER
# ============================================================

class ContextEngineer:
    """Context injection and entity extraction."""
    
    COHORT_KW = ["champions", "loyal", "at risk", "about to sleep", "potential loyalist", "recent", "need attention", "promising", "hibernating", "lost", "price sensitive"]
    METRIC_KW = ["revenue", "monetary", "frequency", "recency", "rfm", "count", "average", "total", "priority", "value"]
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
            text_lower = text.lower()
            # FIX: Tokenize to prevent substring errors (e.g. "rate" inside "separate")
            # We replace punctuation with spaces to handle "revenue?" or "Champions!"
            clean_text = re.sub(r'[^\w\s]', ' ', text_lower)
            tokens = set(clean_text.split())

            found_cohorts = []
            for kw in self.COHORT_KW:
                # Handle multi-word keywords like "at risk"
                if " " in kw:
                    if kw in text_lower: # Keep substring check for phrases
                        found_cohorts.append(kw.title())
                # Handle single-word keywords strictly
                elif kw in tokens:
                    found_cohorts.append(kw.title())

            found_metrics = []
            for kw in self.METRIC_KW:
                if kw in tokens:
                    found_metrics.append(kw)

            return {
                "cohorts": found_cohorts,
                "metrics": found_metrics
            }
        
    def build_context_prompt(self, context: Dict, user_memories: List[Dict] = None) -> str:
        parts = []
        if context.get("segments_discussed"):
            parts.append(f"Segments discussed: {', '.join(context['segments_discussed'][-5:])}")
        if context.get("facts"):
            parts.append(f"Facts: {'; '.join(context['facts'][-3:])}")
        if user_memories:
            for mem in user_memories[:2]:
                if mem.get("summary"): parts.append(f"Past: {mem['summary']}")
        return "\n## Context\n" + "\n".join(parts) if parts else ""
    
    def summarize(self, context: Dict) -> str:
        segs = context.get("segments_discussed", [])
        return f"Discussed: {', '.join(segs[:3])}" if segs else "General cohort discussion"


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """You are a Marketing Intelligence Assistant.

## Data Sources
1. **RFM Segments** (marketing_playbook.json) - Strategic Layer.
2. **HDBSCAN Clusters** (Snowflake) - Tactical Layer.

## Query Routing Rules
- **Strategic Questions** (e.g., "How are Champions doing?") → Use `rfm_segment`.
- **Targeting/Behavior** (e.g., "Who buys on weekends?") → Use `hdbscan_cluster`.
- **Anomalies** → Use BOTH.

## CRITICAL INSTRUCTION: HANDLING CONTEXT & PRONOUNS
You have a memory of **Segments Discussed** (see Context below).
**If the user uses pronouns** like "it", "them", "the first one", or "that group":
1. LOOK at the `Segments discussed` list in the Context.
2. RESOLVE the pronoun to the specific Segment Name.
3. CALL `get_segment_details(segment_name="...")` instead of `smart_customer_query`.

*Example:*
Context: [Champions, Loyal]
User: "Tell me more about the first one."
Action: call `get_segment_details(segment_name="Champions")`
(DO NOT send "first one" to the tool).

## Guidelines
1. **The Safety Net:** If a customer is 'Champions' but 'Cluster -1' (Noise), prioritize RFM but flag for review.
2. **Actionability:** Always quote the specific KPI or Action from the playbook.

## Context
{context}
"""
# ============================================================
# TOOL DECLARATIONS
# ============================================================

TOOLS = [
    {"name": "smart_customer_query", 
     # Explicitly forbid using this for specific details
     "description": "Query customer data. Auto-routes to RFM/HDBSCAN/Both.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "get_segment_details", 
     # Explicitly capture "Tell me more" intents
     "description": "Get segment details from playbook.",
     "parameters": {"type": "object", "properties": {"segment_name": {"type": "string"}}, "required": ["segment_name"]}},
    {"name": "find_behavioral_cluster", "description": "Find customers by category (Snowflake).",
     "parameters": {"type": "object", "properties": {"category": {"type": "string"}}, "required": ["category"]}},
    {"name": "detect_customer_anomalies", "description": "Find high-value anomalies (both sources).",
     "parameters": {"type": "object", "properties": {}, "required": []}},
    {"name": "compare_segments", "description": "Compare segments.",
     "parameters": {"type": "object", "properties": {"segment_names": {"type": "array", "items": {"type": "string"}}}, "required": ["segment_names"]}}
]


# ============================================================
# TOOL EXECUTOR
# ============================================================

class ToolExecutor:
    def __init__(self, tools: CohortTools):
        self.tools = tools
    
    def execute(self, name: str, args: Dict) -> Dict:
        logger.info(f"Executing tool: {name} with args: {args}")
        try:
            if name == "smart_customer_query": return self.tools.smart_query(args.get("query", ""))
            if name == "get_segment_details": return self.tools.get_segment_details(args.get("segment_name", "Champions"))
            if name == "find_behavioral_cluster":   
                category = args.get("category")
                
                # Don't guess . Force the Agent to clarify.
                if not category or not category.strip():
                    return {"error": "Missing parameter: 'category'. Please ask the user which product category they are interested in."}
                
                return self.tools.smart_query(f"find {category} buyers for campaign")
            if name == "detect_customer_anomalies": return self.tools.smart_query("high-value customers acting weirdly")
            if name == "compare_segments": 
                seg_names = args.get("segment_names", [])
                if isinstance(seg_names, str):
                    seg_names = [seg_names]
                if not seg_names:  # Empty list check
                    return {"error": "No segments provided for comparison"}
                return self.tools.compare_segments(seg_names)
            
            logger.warning(f"Unknown tool requested: {name}")
            return {"error": f"Unknown tool: {name}"}
        except Exception as e:
            logger.error(f"Tool {name} failed: {e}", exc_info=True)
            return {"error": str(e)}


# ============================================================
# MARKETING AGENT
# ============================================================

class MarketingAgent:
    def __init__(self, config: AgentConfig = None):
        self.config = config or AgentConfig()
        self.memory = MemoryManager(self.config.memory_db_path)
        self.context_eng = ContextEngineer()
        
        sf_config = {'account': self.config.snowflake_account, 'user': self.config.snowflake_user,
                     'password': self.config.snowflake_password} if self.config.use_snowflake else None
        self.cohort_tools = CohortTools(self.config.playbook_path, sf_config, self.config.use_snowflake)
        self.tool_exec = ToolExecutor(self.cohort_tools)
        
        self.client = None
        if ADK_AVAILABLE and self.config.gemini_api_key:
            self.client = genai.Client(api_key=self.config.gemini_api_key)
    
    def start_conversation(self, user_id: str = "default") -> str:
        return self.memory.create_conversation(user_id)
    
    def _resolve_query(self, query: str, context: Dict) -> Tuple[Optional[str], Optional[str], Optional[Dict]]:
        """
        Pre-process query to resolve pronouns and detect special query types.
        Returns: (tool_to_use, resolved_segment, tool_args) or (None, None, None) if no resolution needed.
        
        This runs BEFORE Gemini to ensure consistent behavior.
        """
        segments_discussed = context.get("segments_discussed", [])
        query_lower = query.lower()
        
        # 1. Check for COMPARISON queries
        comparison_keywords = ["compare", "versus", "vs", "difference between", "comparison"]
        if any(kw in query_lower for kw in comparison_keywords):
            segment_names = []
            all_segments = ["champions", "loyal", "at risk", "about to sleep", "potential loyalist", 
                           "recent", "need attention", "promising", "hibernating", "lost", "price sensitive"]
            for seg in all_segments:
                if seg in query_lower:
                    segment_names.append(seg.title())
            
            if len(segment_names) >= 2:
                logger.info(f"Comparison query detected. Comparing: {segment_names}")
                return "compare_segments", None, {"segment_names": segment_names}
        
        # 2. Check for pronoun references that need context resolution
        pronoun_patterns = {
            "first one": 0, "the first": 0,
            "second one": 1, "the second": 1,
            "third one": 2, "the third": 2,
            "that one": 0, "this one": 0,
            "that segment": 0, "this segment": 0,
            "tell me more": 0, "more about it": 0, "more details": 0,
            "expand on": 0, "elaborate": 0,
        }
        
        for pattern, index in pronoun_patterns.items():
            if pattern in query_lower and segments_discussed:
                if index < len(segments_discussed):
                    resolved_segment = segments_discussed[index]
                    logger.info(f"Resolved '{pattern}' to segment: {resolved_segment}")
                    return "get_segment_details", resolved_segment, {"segment_name": resolved_segment}
        
        # 3. Check for prioritization/recommendation queries
        priority_keywords = ["prioritize", "priority", "focus on", "should we", "recommend", "suggestion"]
        if any(kw in query_lower for kw in priority_keywords):
            return "smart_customer_query", None, {"query": "top priority segments with actions"}
        
        # No special resolution needed
        return None, None, None
    
    def chat(self, query: str, conv_id: str = None, user_id: str = "default") -> Dict:
        if not conv_id or not self.memory.exists(conv_id):
            conv_id = self.start_conversation(user_id)
        
        self.memory.add_message(conv_id, "user", query)
        context = self.memory.get_context(conv_id)
        messages = self.memory.get_messages(conv_id, self.config.max_messages)
        memories = self.memory.get_user_memories(user_id, 3)
        
        analysis = detect_query_intent(query)
        entities = self.context_eng.extract_entities(query)
        self.memory.update_context(conv_id, segments_discussed=entities["cohorts"],
                                   last_intent=analysis.intent.value, last_data_source=analysis.column_selection.value)
        
        # PRE-PROCESS: Resolve pronouns and special queries BEFORE calling Gemini
        resolved_tool, resolved_segment, resolved_args = self._resolve_query(query, context)
        
        if resolved_tool:
            # We have a resolved query - execute directly without Gemini
            result = self._execute_resolved(resolved_tool, resolved_args, analysis, resolved_segment)
        elif self.client:
            # No resolution needed - let Gemini handle it
            result = self._respond(query, analysis, context, messages, memories)
        else:
            # No Gemini client - use fallback
            result = self._fallback(query, analysis, context)
        
        resp_entities = self.context_eng.extract_entities(result.get("response", ""))
        self.memory.update_context(conv_id, segments_discussed=resp_entities["cohorts"])
        self.memory.add_message(conv_id, "assistant", result.get("response", ""))
        
        ctx = self.memory.get_context(conv_id)
        self.memory.save_long_term(user_id, conv_id, self.context_eng.summarize(ctx), ctx.get("facts", []))
        
        result.update({"conversation_id": conv_id, "message_count": len(messages) + 2,
                       "context": {"segments": ctx.get("segments_discussed", [])[-5:], "facts": ctx.get("facts", [])[-3:]}})
        return result
    
    def _execute_resolved(self, tool_name: str, tool_args: Dict, analysis, resolved_segment: str = None) -> Dict:
        """Execute a pre-resolved tool call (bypasses Gemini)."""
        result = self.tool_exec.execute(tool_name, tool_args)
        
        return {
            "response": self._format([{"tool": tool_name, "result": result}], analysis),
            "tools_used": [tool_name],
            "tool_results": [{"tool": tool_name, "result": result}],
            "query_analysis": {
                "intent": analysis.intent.value,
                "column_selection": analysis.column_selection.value,
                "resolved_segment": resolved_segment
            },
            "pre_resolved": True
        }
    
    def _respond(self, query: str, analysis, context: Dict, messages: List[Dict], memories: List[Dict]) -> Dict:
        ctx_prompt = self.context_eng.build_context_prompt(context, memories)
        system = SYSTEM_PROMPT.format(context=ctx_prompt)
        
        try:
            tools_cfg = types.Tool(function_declarations=[
                types.FunctionDeclaration(name=t["name"], description=t["description"], parameters=t["parameters"]) for t in TOOLS
            ])
            
            # Build properly formatted content objects for Gemini API
            formatted_contents = [
                types.Content(
                    role=m["role"], 
                    parts=[types.Part(text=m["content"])]
                ) for m in messages[-10:]
            ]
            
            resp = self.client.models.generate_content(
                model=self.config.model_name,
                contents=formatted_contents,
                config=types.GenerateContentConfig(system_instruction=system, tools=[tools_cfg], temperature=self.config.temperature)
            )
            
            tools_used, tool_results, final = [], [], ""
            for cand in resp.candidates:
                for part in cand.content.parts:
                    if hasattr(part, 'function_call') and part.function_call:
                        tools_used.append(part.function_call.name)
                        tool_results.append({"tool": part.function_call.name,
                                            "result": self.tool_exec.execute(part.function_call.name, dict(part.function_call.args) if part.function_call.args else {})})
                    if hasattr(part, 'text') and part.text: final = part.text
            
            if tool_results and not final: final = self._format(tool_results, analysis)
            return {"response": final, "tools_used": tools_used, "tool_results": tool_results,
                    "query_analysis": {"intent": analysis.intent.value, "column_selection": analysis.column_selection.value, "reasoning": analysis.reasoning}}
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return self._fallback(query, analysis, context)
    
    def _fallback(self, query: str, analysis, context: Dict = None) -> Dict:
        """
        Fallback response when Gemini API is unavailable.
        Uses the same resolution logic as _resolve_query.
        """
        context = context or {}
        
        # Use the shared resolution logic
        resolved_tool, resolved_segment, resolved_args = self._resolve_query(query, context)
        
        if resolved_tool:
            tool_to_use = resolved_tool
            tool_args = resolved_args
        else:
            tool_to_use = "smart_customer_query"
            tool_args = {"query": query}
        
        result = self.tool_exec.execute(tool_to_use, tool_args)
        
        return {
            "response": self._format([{"tool": tool_to_use, "result": result}], analysis),
            "tools_used": [tool_to_use], 
            "tool_results": [{"tool": tool_to_use, "result": result}],
            "query_analysis": {
                "intent": analysis.intent.value, 
                "column_selection": analysis.column_selection.value,
                "resolved_segment": resolved_segment
            }, 
            "fallback": True
        }
    
    def _format(self, tool_results: List[Dict], analysis) -> str:
        parts = [f"**Data Source:** {analysis.column_selection.value}", f"**Reasoning:** {analysis.reasoning}\n"]
        for tr in tool_results:
            r = tr["result"]
            
            if "error" in r: 
                parts.append(f"Error: {r['error']}")
                continue
            
            # Handle segment list (from smart_customer_query)
            if "segments" in r:
                parts.append("**Segments:**")
                for s in r["segments"][:5]:
                    parts.append(f"- {s.get('rfm_segment', 'N/A')}: {s.get('customer_count', 'N/A')} customers, ${s.get('avg_monetary', 0):.0f} avg")
                    if s.get('marketing_goal'): 
                        parts.append(f"  Goal: {s['marketing_goal']}")
                    if s.get('recommended_actions'): 
                        actions_preview = ', '.join(s['recommended_actions'][:2])[:60]
                        parts.append(f"  Actions: {actions_preview}...")
            
            # Handle single segment details (from get_segment_details)
            if "segment" in r and "segments" not in r:
                s = r["segment"]
                parts.append(f"\n**Segment: {s['rfm_segment']}**")
                parts.append(f"- Priority: {s.get('priority', 'N/A')}")
                parts.append(f"- Customers: {s.get('customer_count', 'N/A'):,}")
                parts.append(f"- Avg Value: ${s.get('avg_monetary', 0):,.0f}")
                if s.get('description'):
                    parts.append(f"- Description: {s['description']}")
                if s.get('marketing_goal'): 
                    parts.append(f"- Goal: {s['marketing_goal']}")
                if s.get('budget_allocation'):
                    parts.append(f"- Budget: {s['budget_allocation']}")
                if s.get('recommended_actions'):
                    parts.append("\n**Recommended Actions:**")
                    for i, a in enumerate(s['recommended_actions'][:5], 1): 
                        parts.append(f"  {i}. {a}")
                if s.get('kpis'):
                    parts.append("\n**Key KPIs:**")
                    for kpi in s['kpis'][:3]:
                        parts.append(f"  • {kpi}")
            
            # Handle anomalies
            if "anomalies" in r:
                parts.append(f"\n**Anomalies Detected:** {r.get('total_anomalies', 0)}")
                if r.get('anomaly_type'):
                    parts.append(f"Type: {r['anomaly_type']}")
            
            # Handle comparison
            if r.get('type') == 'segment_comparison' and 'insights' in r:
                parts.append("\n**Comparison Insights:**")
                for insight in r['insights']:
                    parts.append(f"• {insight}")
                    
        return "\n".join(parts)
    
    def get_history(self, conv_id: str) -> List[Dict]: return self.memory.get_messages(conv_id)
    def get_context(self, conv_id: str) -> Dict: return self.memory.get_context(conv_id)


# ============================================================
# CLI
# ============================================================

def run_interactive():
    print("\n" + "="*60 + "\nMARKETING INTELLIGENCE AGENT\nADK + Gemini + Memory\n" + "="*60)
    print("\nData: marketing_playbook.json + Snowflake")
    print("Commands: /history /context /new /quit\n")
    
    agent = MarketingAgent(AgentConfig(gemini_api_key=os.getenv("GEMINI_API_KEY", "")))
    conv_id = agent.start_conversation()
    print(f"📝 Conversation: {conv_id}\n")
    
    while True:
        try:
            q = input("👤 You: ").strip()
            if not q: continue
            if q == "/quit": break
            if q == "/new": conv_id = agent.start_conversation(); print(f"📝 New: {conv_id}"); continue
            if q == "/history":
                for m in agent.get_history(conv_id)[-6:]: print(f"  {'👤' if m['role']=='user' else '🤖'} {m['content'][:60]}...")
                continue
            if q == "/context":
                c = agent.get_context(conv_id); print(f"  Segments: {c.get('segments_discussed', [])}\n  Facts: {c.get('facts', [])}"); continue
            
            r = agent.chat(q, conv_id)
            print(f"\n[{r.get('query_analysis',{}).get('intent')} → {r.get('query_analysis',{}).get('column_selection')}]")
            print(f"\n🤖 {r['response']}\n[Conv: {r['conversation_id']} | Msgs: {r.get('message_count',0)}]\n")
        except KeyboardInterrupt: break
    print("Goodbye!")


def run_demo():
    print("\n" + "="*60 + "\nDEMO: Memory & Context Resolution in Action\n" + "="*60)
    agent = MarketingAgent()
    conv_id = agent.start_conversation()
    
    queries = [
        ("Show me top 3 segments", "Initial query - returns top 3 segments"),
        ("Tell me more about the first one", "Pronoun resolution - should resolve to 'Champions'"),
        ("What should we prioritize?", "Strategic query - priority-based recommendations"),
        ("Compare Champions and At Risk", "Comparison query - compares both segments"),
    ]
    
    for i, (q, description) in enumerate(queries, 1):
        print(f"\n{'='*40}\nQuery {i}: {q}")
        print(f"Expected: {description}")
        print("-"*40)
        r = agent.chat(q, conv_id)
        
        # Show context tracking
        ctx = r.get('context', {})
        segments = ctx.get('segments', [])
        print(f"📊 Context (segments discussed): {segments}")
        
        # Show if query was pre-resolved
        if r.get('pre_resolved'):
            qa = r.get('query_analysis', {})
            if qa.get('resolved_segment'):
                print(f"✅ Pronoun resolved to: {qa['resolved_segment']}")
            else:
                print(f"✅ Query pre-resolved")
        
        # Show tools used
        print(f"🔧 Tools used: {r.get('tools_used', [])}")
        
        # Show response (truncated)
        response = r['response']
        if len(response) > 500:
            print(f"\n🤖 Response:\n{response[:500]}...")
        else:
            print(f"\n🤖 Response:\n{response}")
    
    print("\n" + "="*60)
    print("DEMO COMPLETE - Memory persists across queries!")
    print("="*60)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--query', type=str)
    p.add_argument('--conversation-id', type=str)
    p.add_argument('--playbook', default='./marketing_playbook.json')
    p.add_argument('--use-snowflake', action='store_true')
    p.add_argument('--demo', action='store_true')
    args = p.parse_args()
   
    # Consistent API Key Loading
    api_key = os.getenv("GEMINI_API_KEY", "")

    if args.demo: run_demo()
    elif args.query:
        agent = MarketingAgent(AgentConfig(playbook_path=args.playbook, use_snowflake=args.use_snowflake))
        r = agent.chat(args.query, args.conversation_id)
        print(r['response'] + f"\n[Conv: {r['conversation_id']}]")
    else: 
        run_interactive()


if __name__ == "__main__": 
    main()