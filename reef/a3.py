import os
import requests
import json
from datetime import datetime
from typing import List, Dict, Optional, Union
from dataclasses import dataclass, field
from bs4 import BeautifulSoup
import uuid

# Configuration
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "Serper-API-key-here")
SEARCH_URL = "https://google.serper.dev/search"

# Data Objects
@dataclass
class Query:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    text: str = ""
    ts: datetime = field(default_factory=datetime.now)

@dataclass
class Source:
    url: str
    type: str  # "api", "web", "pdf", "image", "app_screen"
    access_mode: str  # "direct", "visual"
    fetched_at: datetime = field(default_factory=datetime.now)

@dataclass
class Extraction:
    source_url: str
    modality: str  # "text", "screenshot", "table", "chart"
    content_text: str
    provenance: Dict = field(default_factory=dict)
    confidence: float = 1.0

@dataclass
class Record:
    claim: str
    evidence: List[Extraction] = field(default_factory=list)
    stance: str = "support"  # "support", "refute", "neutral"
    metrics: Dict = field(default_factory=dict)

# Navigator & Extractor Agent
class NavigatorExtractor:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {'X-API-KEY': api_key, 'Content-Type': 'application/json'}
    
    def plan(self, query: Query) -> Dict:
        """LLM Orchestrator: Plan tasks and choose access mode"""
        # Simplified planning logic - in real implementation, use LLM
        return {
            "mode": "direct",  # or "visual"
            "tasks": [
                {"type": "search", "query": query.text, "k": 5},
                {"type": "extract", "urls": []}  # Will be populated after search
            ]
        }
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Search Tool: Web search → ranked URLs"""
        payload = json.dumps({
            "q": f"{query} site:kubernetes.io OR site:github.com/kubernetes",
            "num": k
        })
        
        try:
            response = requests.post(SEARCH_URL, headers=self.headers, data=payload)
            response.raise_for_status()
            results = response.json().get('organic', [])
            return [{"url": r.get("link"), "title": r.get("title"), "snippet": r.get("snippet")} 
                    for r in results]
        except Exception as e:
            print(f"Search error: {e}")
            return []
    
    def fetch_text(self, url: str) -> Optional[str]:
        """Text Extractor Tool: Fetch + clean text from URLs"""
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove non-content elements
            for element in soup(["script", "style", "nav", "footer"]):
                element.decompose()
            
            # Extract main content
            content = soup.find('main') or soup.find('article') or soup
            return content.get_text(separator=' ', strip=True)[:2000]  # Limit length
        except Exception as e:
            print(f"Extraction error: {e}")
            return None
    
    def execute_plan(self, plan: Dict, query: Query) -> List[Record]:
        """Execute the planned tasks"""
        records = []
        
        # Execute search task
        search_results = []
        for task in plan.get("tasks", []):
            if task["type"] == "search":
                search_results = self.search(task["query"], task.get("k", 5))
                # Update extract task with found URLs
                for t in plan["tasks"]:
                    if t["type"] == "extract":
                        t["urls"] = [r["url"] for r in search_results]
        
        # Execute extract tasks
        for task in plan.get("tasks", []):
            if task["type"] == "extract":
                for url in task.get("urls", []):
                    text = self.fetch_text(url)
                    if text:
                        extraction = Extraction(
                            source_url=url,
                            modality="text",
                            content_text=text,
                            provenance={"url": url}
                        )
                        record = Record(
                            claim=f"Information about {query.text}",
                            evidence=[extraction]
                        )
                        records.append(record)
        
        return records

# Aggregator Service
class Aggregator:
    def __init__(self):
        self.kb_store = {}  # Simplified KB store
        self.claims_index = {}  # For quick lookup
    
    def normalize(self, records: List[Record]) -> List[Record]:
        """Normalizer: De-dupe, chunk, and canonicalize facts"""
        # Simplified normalization
        return records
    
    def aggregate(self, op: str, records: List[Record]) -> Dict:
        """Fusion Engine: Merge or replace entries"""
        result = {"kb_ids": [], "status": "success"}
        
        for record in records:
            claim_hash = hash(record.claim)
            
            if op == "ADD":
                if claim_hash not in self.claims_index:
                    # New claim
                    record_id = str(uuid.uuid4())
                    self.kb_store[record_id] = record
                    self.claims_index[claim_hash] = record_id
                    result["kb_ids"].append(record_id)
                else:
                    # Existing claim - add evidence
                    record_id = self.claims_index[claim_hash]
                    self.kb_store[record_id].evidence.extend(record.evidence)
                    result["kb_ids"].append(record_id)
            
            elif op == "REPLACE":
                if claim_hash in self.claims_index:
                    record_id = self.claims_index[claim_hash]
                    self.kb_store[record_id] = record
                    result["kb_ids"].append(record_id)
        
        return result
    
    def generate_feedback(self, kb_state: Dict, query: Query) -> Dict:
        """Feedback Generator: Gaps, conflicts, and next-best links"""
        # Simplified feedback
        return {
            "gaps": ["More recent Kubernetes versions", "Production deployment examples"],
            "conflicts": [],
            "suggestions": ["Check official Kubernetes documentation", "Review GitHub examples"]
        }

# Main System
class KubernetesSearchSystem:
    def __init__(self):
        self.navigator = NavigatorExtractor(SERPER_API_KEY)
        self.aggregator = Aggregator()
        self.max_pages = 10  # Quality guardrail: max pages per query
    
    def ask(self, query_text: str) -> Dict:
        """Main entry point for user queries"""
        query = Query(text=query_text)
        
        # Plan and execute
        plan = self.navigator.plan(query)
        records = self.navigator.execute_plan(plan, query)
        
        # Normalize and aggregate
        normalized_records = self.aggregator.normalize(records)
        agg_result = self.aggregator.aggregate("ADD", normalized_records)
        
        # Generate feedback
        feedback = self.aggregator.generate_feedback(self.aggregator.kb_store, query)
        
        # Check if we need more iterations (simplified)
        if feedback["gaps"] and len(records) < self.max_pages:
            # Plan next iteration based on feedback
            next_query = Query(text=f"{query.text} {feedback['suggestions'][0]}")
            next_plan = self.navigator.plan(next_query)
            next_records = self.navigator.execute_plan(next_plan, next_query)
            
            # Aggregate new records
            next_normalized = self.aggregator.normalize(next_records)
            self.aggregator.aggregate("ADD", next_normalized)
        
        # Compose answer
        answer = self.compose_answer(query)
        
        return {
            "query": query.text,
            "answer": answer,
            "sources": [e.source_url for r in records for e in r.evidence],
            "feedback": feedback
        }
    
    def compose_answer(self, query: Query) -> str:
        """Generate a grounded response citing KB entries"""
        # Simplified answer composition
        relevant_records = [
            record for record in self.aggregator.kb_store.values()
            if query.text.lower() in record.claim.lower()
        ]
        
        if not relevant_records:
            return "No relevant information found in the knowledge base."
        
        answer = f"Based on {len(relevant_records)} sources:\n\n"
        for record in relevant_records[:3]:  # Limit to top 3
            answer += f"- {record.claim}\n"
            for evidence in record.evidence[:1]:  # Show one evidence source
                answer += f"  Source: {evidence.source_url}\n"
        
        return answer

# Usage Example
if __name__ == "__main__":
    system = KubernetesSearchSystem()
    
    # User query
    user_query = "How to deploy a stateful application on Kubernetes?"
    
    # Execute search
    result = system.ask(user_query)
    
    # Display results
    print("="*50)
    print(f"QUERY: {result['query']}")
    print("="*50)
    print("\nANSWER:")
    print(result['answer'])
    
    print("\nSOURCES:")
    for i, source in enumerate(result['sources'], 1):
        print(f"{i}. {source}")
    
    print("\nFEEDBACK:")
    print(f"Gaps: {', '.join(result['feedback']['gaps'])}")
    print(f"Suggestions: {', '.join(result['feedback']['suggestions'])}")