#!/usr/bin/env python3
"""
MARKETING INTELLIGENCE AGENT 
==================================================
- tools:
    smart_customer_query,
    get_segment_details,
    find_behavioral_cluster,
    detect_customer_anomalies,
    compare_segments
- system instruction
- model
"""

import asyncio
import os
import sys
import logging
import uuid
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("marketing_agent")

# Local imports
sys.path.insert(0, str(Path(__file__).parent))

from cohort_tools import (
    initialize_data_loader,
    smart_customer_query,
    get_segment_details,
    find_behavioral_cluster,
    detect_customer_anomalies,
    compare_segments
)

# ============================================================
# GOOGLE ADK IMPORTS
# ============================================================
try:
    from google.adk.agents import LlmAgent
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai.types import Content, Part
    ADK_AVAILABLE = True
except ImportError as e:
    logger.error(f"Google ADK import failed: {e}")
    logger.error("Please install: pip install google-adk")
    sys.exit(1)

# ============================================================
# CONSTANTS & CONFIG
# ============================================================
APP_NAME = "marketing_intelligence"
MODEL_NAME = "gemini-2.5-flash"

# Initialize Data Layer
initialize_data_loader(
    playbook_path=os.getenv("PLAYBOOK_PATH", "./marketing_playbook.json"),
    use_snowflake=os.getenv("USE_SNOWFLAKE", "false").lower() == "true"
)

# ============================================================
# SYSTEM INSTRUCTION
# ============================================================
INSTRUCTION = """You are a Marketing Intelligence Assistant.

## Data Sources
1. **RFM Segments** (Strategic Layer): Use for high-level overviews, health checks, and segment details.
2. **HDBSCAN Clusters** (Tactical Layer): Use for behavioral targeting (e.g., "who buys electronics").

## Tool Usage Guidelines
- **Strategic Questions** ("How are Champions?", "Top segments") -> `smart_customer_query`
- **Specific Details** ("Tell me about Champions", "Deep dive on them") -> `get_segment_details`
- **Comparisons** ("Compare Champions and Lost") -> `compare_segments`
- **Targeting** ("Who buys tech?") -> `find_behavioral_cluster`
- **Anomalies** ("Find weird behavior") -> `detect_customer_anomalies`

## Context Awareness
- You have access to the conversation history.
- If the user says "Tell me more about **it**" or "**that one**", infer the segment from the previous messages.
- Do not ask the user for clarification if the segment is obvious from the last few turns.

## Response Style
- Be concise and actionable.
- Always quote specific KPIs or Actions when available.
"""

# ============================================================
# AGENT DEFINITION
# ============================================================

marketing_agent = LlmAgent(
    name="marketing_agent",
    model=MODEL_NAME,
    description="Analyzes customer segments and marketing data.",
    instruction=INSTRUCTION,
    tools=[
        smart_customer_query,
        get_segment_details,
        find_behavioral_cluster,
        detect_customer_anomalies,
        compare_segments,
    ],
    # ADD THIS: Automatically saves the final response to session.state["last_agent_response"]
    output_key="last_agent_response"
)

# ============================================================
# MAIN CLASS
# ============================================================

class MarketingAgentService:
    def __init__(self):
        self.session_service = InMemorySessionService()
        self.runner = Runner(
            agent=marketing_agent,
            app_name=APP_NAME,
            session_service=self.session_service
        )
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def _run_async(self, coro):
        return self.loop.run_until_complete(coro)

    def start_conversation(self, user_id: str = "default") -> str:
        session_id = str(uuid.uuid4())[:8]
        self._run_async(
            self.session_service.create_session(
                app_name=APP_NAME,
                session_id=session_id,
                user_id=user_id,
                state={"turn_count": 0}
            )
        )
        return session_id

    def chat(self, query: str, session_id: str = None, user_id: str = "default") -> Dict[str, Any]:
        if not session_id:
            session_id = self.start_conversation(user_id)

        # Execute Runner
        response_text = ""
        tools_used = []
        
        user_msg = Content(role="user", parts=[Part(text=query)])

        events = self.runner.run(
            user_id=user_id,
            session_id=session_id,
            new_message=user_msg
        )

        for event in events:
            if hasattr(event, "function_calls") and event.function_calls:
                for fc in event.function_calls:
                    tools_used.append(fc.name)
            
            if event.is_final_response() and event.content and event.content.parts:
                response_text = event.content.parts[0].text

        return {
            "response": response_text,
            "session_id": session_id,
            "tools_used": tools_used
        }

# ============================================================
# CLI & DEMO
# ============================================================

def format_output(response_data: Dict):
    print(f"\n🤖 Agent: {response_data['response']}")
    if response_data['tools_used']:
        print(f"   (Tools: {', '.join(response_data['tools_used'])})")
    print("-" * 60)

def run_demo():
    print("=" * 60)
    print("MARKETING AGENT - ADK DEMO")
    print("Focus: Context Retention & Intelligent Routing")
    print("=" * 60)

    agent = MarketingAgentService()
    sid = agent.start_conversation()

    scenarios = [
        "Show me the top 3 segments",
        "Tell me more about the first one",
        "Compare it with the Lost segment",
        "What should we prioritize?"
    ]

    for q in scenarios:
        print(f"\n👤 User: {q}")
        print("   (Thinking... waiting for API rate limit)")
        time.sleep(20) # Keep this safely high for Free Tier
        
        try:
            result = agent.chat(q, session_id=sid)
            format_output(result)
        except Exception as e:
            print(f"❌ Error: {e}")
            break

def run_interactive():
    agent = MarketingAgentService()
    sid = agent.start_conversation()
    print("\nStarting Interactive Session (Type 'quit' to exit)")
    print("-" * 60)
    
    while True:
        try:
            q = input("\n👤 You: ").strip()
            if q.lower() in ['quit', 'exit']:
                break
            result = agent.chat(q, session_id=sid)
            format_output(result)
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="Run automated demo")
    args = parser.parse_args()

    if args.demo:
        run_demo()
    else:
        run_interactive()