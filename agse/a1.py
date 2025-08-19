import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import re
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Load environment variables
load_dotenv()

class NRPKnowledgeBase:
    def __init__(self):
        self.api_key = os.getenv("NRP_API_KEY", "NRP-API-key-here")
        self.base_url = "https://llm.nrp-nautilus.io/"
        self.embedding_endpoint = f"{self.base_url}/v1/embeddings"
        self.rerank_endpoint = f"{self.base_url}/v1/rerank"
        
        # Initialize knowledge base storage
        self.documents = []
        self.embeddings = None
        self.metadata = []
        
        # Create a robust session with retries
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set headers to identify our crawler
        self.session.headers.update({
            "User-Agent": "NRP-Documentation-Crawler/1.0"
        })
        
    def crawl_documentation(self, base_url, max_depth=2, delay=2, timeout=30):
        """Crawl the NRP.ai documentation with improved error handling"""
        visited_urls = set()
        pages_to_crawl = [(base_url, 0)]
        failed_urls = []
        
        while pages_to_crawl:
            url, depth = pages_to_crawl.pop(0)
            
            if depth > max_depth or url in visited_urls:
                continue
                
            visited_urls.add(url)
            time.sleep(delay)  # Increased delay between requests
            
            try:
                print(f"Crawling: {url} (depth: {depth})")
                response = self.session.get(url, timeout=timeout)
                
                if response.status_code != 200:
                    print(f"Failed to fetch {url}: Status {response.status_code}")
                    failed_urls.append(url)
                    continue
                    
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract page content
                title = soup.find('title').get_text() if soup.find('title') else "No Title"
                
                # Try to find the main content area
                content_div = soup.find('div', class_='documentation') or \
                              soup.find('main') or \
                              soup.find('article') or \
                              soup.find('div', class_='content') or \
                              soup
                
                content = content_div.get_text(strip=True)
                
                # Skip pages with very little content
                if len(content) < 100:
                    print(f"Skipping {url} - insufficient content")
                    continue
                
                # Store document with metadata
                self.documents.append({
                    'text': content,
                    'url': url,
                    'title': title
                })
                
                # Find all links
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    full_url = urljoin(url, href)
                    
                    # Only follow links within the documentation
                    if urlparse(full_url).netloc == urlparse(base_url).netloc:
                        pages_to_crawl.append((full_url, depth + 1))
                        
            except requests.exceptions.RequestException as e:
                print(f"Error crawling {url}: {e}")
                failed_urls.append(url)
                continue
                
        print(f"Crawled {len(self.documents)} pages")
        if failed_urls:
            print(f"Failed to crawl {len(failed_urls)} pages:")
            for url in failed_urls:
                print(f"  - {url}")
                
    def get_embeddings(self, texts, batch_size=10):
        """Get embeddings from the NRP API with batching"""
        all_embeddings = []
        
        # Process texts in batches to avoid overwhelming the API
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "embed-mistral",
                "input": batch
            }
            
            try:
                response = self.session.post(self.embedding_endpoint, json=data, headers=headers, timeout=30)
                if response.status_code == 200:
                    result = response.json()
                    all_embeddings.extend([item['embedding'] for item in result['data']])
                    print(f"Processed batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
                else:
                    print(f"Error getting embeddings: {response.status_code} - {response.text}")
                    # Add zero embeddings as fallback
                    all_embeddings.extend([[0.0] * 768] * len(batch))
            except Exception as e:
                print(f"Exception when getting embeddings: {e}")
                # Add zero embeddings as fallback
                all_embeddings.extend([[0.0] * 768] * len(batch))
                
            # Add delay between batches
            time.sleep(1)
            
        return all_embeddings
    
    def rerank_results(self, query, documents, top_k=5):
        """Rerank search results using the NRP API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Prepare documents for reranking
        docs_for_rerank = [{"text": doc['text']} for doc in documents]
        
        data = {
            "model": "gemma3",
            "query": query,
            "documents": docs_for_rerank,
            "top_n": top_k
        }
        
        try:
            response = self.session.post(self.rerank_endpoint, json=data, headers=headers, timeout=30)
            if response.status_code == 200:
                result = response.json()
                # Get indices of top results
                top_indices = [item['index'] for item in result['results']]
                return [documents[i] for i in top_indices]
            else:
                print(f"Error reranking results: {response.status_code} - {response.text}")
                return documents[:top_k]  # Fallback to original order
        except Exception as e:
            print(f"Exception when reranking: {e}")
            return documents[:top_k]  # Fallback to original order
    
    def build_knowledge_base(self):
        """Build the knowledge base with embeddings"""
        if not self.documents:
            print("No documents to process. Please crawl the documentation first.")
            return
            
        # Get embeddings for all documents
        print("Generating embeddings...")
        texts = [doc['text'] for doc in self.documents]
        embeddings = self.get_embeddings(texts)
        
        if embeddings is None:
            print("Failed to generate embeddings")
            return
            
        self.embeddings = np.array(embeddings)
        self.metadata = [{
            'url': doc['url'],
            'title': doc['title']
        } for doc in self.documents]
        
        print(f"Knowledge base built with {len(self.documents)} documents")
    
    def search(self, query, top_k=5, use_reranking=True):
        """Search the knowledge base"""
        if self.embeddings is None:
            print("Knowledge base not built. Please build it first.")
            return []
            
        # Get query embedding
        query_embedding = self.get_embeddings([query])
        if query_embedding is None:
            print("Failed to generate query embedding")
            return []
            
        query_embedding = np.array(query_embedding[0]).reshape(1, -1)
        
        # Calculate similarity
        similarities = cosine_similarity(query_embedding, self.embeddings).flatten()
        
        # Get top results
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        # Prepare results
        results = []
        for idx in top_indices:
            results.append({
                'text': self.documents[idx]['text'],
                'url': self.metadata[idx]['url'],
                'title': self.metadata[idx]['title'],
                'score': float(similarities[idx])
            })
        
        # Apply reranking if requested
        if use_reranking and len(results) > 0:
            results = self.rerank_results(query, results, top_k)
            
        return results
    
    def save_knowledge_base(self, filepath):
        """Save the knowledge base to disk"""
        if self.embeddings is None:
            print("Knowledge base not built. Nothing to save.")
            return
            
        data = {
            'documents': self.documents,
            'embeddings': self.embeddings.tolist(),
            'metadata': self.metadata
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f)
            
        print(f"Knowledge base saved to {filepath}")
    
    def load_knowledge_base(self, filepath):
        """Load the knowledge base from disk"""
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        self.documents = data['documents']
        self.embeddings = np.array(data['embeddings'])
        self.metadata = data['metadata']
        
        print(f"Knowledge base loaded from {filepath} with {len(self.documents)} documents")

# Create and initialize the knowledge base
nrp_kb = NRPKnowledgeBase()

# Crawl the documentation with improved settings
print("Crawling NRP.ai documentation...")
nrp_kb.crawl_documentation(
    "https://nrp.ai/documentation/", 
    max_depth=2, 
    delay=3,  # Increased delay
    timeout=30  # Increased timeout
)

# Build the knowledge base
print("Building knowledge base...")
nrp_kb.build_knowledge_base()

# Save the knowledge base for future use
nrp_kb.save_knowledge_base("nrp_expert_docs_kb.json")

# Example search
print("\nExample search for 'GPU pods':")
results = nrp_kb.search("GPU pods", top_k=3)
for result in results:
    print(f"\nTitle: {result['title']}")
    print(f"URL: {result['url']}")
    print(f"Score: {result['score']}")
    print(f"Content: {result['text'][:200]}...")