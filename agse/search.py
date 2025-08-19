# search.py - Enhanced with LLM capabilities
from elasticsearch import Elasticsearch
from llm_utils import get_embedding, find_cross_reference, get_llm_response
import numpy as np

class DocumentationSearch:
    def __init__(self):
        self.es = Elasticsearch(['http://localhost:9200'])
    
    def search(self, query, top_n=10):
        # Step 1: Understand the query with LLM
        query_analysis = self.analyze_query(query)
        
        # Step 2: Get embedding for the query
        query_embedding = get_embedding(query)
        
        # Step 3: Perform vector search
        results = self.vector_search(query_embedding, top_n)
        
        # Step 4: Enhance results with LLM
        enhanced_results = self.enhance_results(query, results)
        
        return enhanced_results
    
    def analyze_query(self, query):
        """Use LLM to understand the query intent and extract key concepts"""
        prompt = f"""
        Analyze the following search query about NRP and Kubernetes:
        
        Query: {query}
        
        Provide:
        1. The main intent of the query
        2. Key technical concepts mentioned
        3. Whether the user is looking for NRP-specific information, Kubernetes context, or both
        """
        messages = [
            {"role": "developer", "content": "You are an expert in cloud-native platforms and Kubernetes."},
            {"role": "user", "content": prompt}
        ]
        return get_llm_response(messages)
    
    def vector_search(self, query_embedding, top_n):
        """Perform vector search using Elasticsearch"""
        script_query = {
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                        "params": {"query_vector": query_embedding}
                    }
                }
            },
            "size": top_n * 2  # Get more to allow for filtering
        }
        
        response = self.es.search(index="documentation", body=script_query)
        hits = response['hits']['hits']
        results = []
        for hit in hits:
            source = hit['_source']
            results.append({
                'url': source['url'],
                'source': source['source'],
                'content': source.get('content', ''),
                'summary': source.get('summary', ''),
                'key_concepts': source.get('key_concepts', []),
                'score': hit['_score']
            })
        return results
    
    def enhance_results(self, query, results):
        """Enhance search results with LLM-generated insights and cross-references"""
        # Group results by source
        nrp_results = [r for r in results if r['source'] == 'nrp']
        k8s_results = [r for r in results if r['source'] == 'k8s']
        
        # For each NRP result, find relevant Kubernetes cross-references
        for nrp_result in nrp_results:
            if k8s_results:
                # Find the most relevant Kubernetes document
                best_k8s_match = max(k8s_results, key=lambda k: len(set(nrp_result['key_concepts']) & set(k['key_concepts'])))
                
                # Generate cross-reference explanation
                cross_ref = find_cross_reference(
                    nrp_result['summary'],
                    best_k8s_match['summary']
                )
                
                nrp_result['cross_reference'] = {
                    'url': best_k8s_match['url'],
                    'explanation': cross_ref,
                    'relevance_score': len(set(nrp_result['key_concepts']) & set(best_k8s_match['key_concepts']))
                }
        
        # Sort results by relevance and prioritize NRP content
        enhanced_results = sorted(
            results,
            key=lambda x: (x['source'] != 'nrp', -x['score']),
        )
        
        return enhanced_results[:10]  # Return top 10 results