# crawler.py - Enhanced with LLM processing
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import redis
import json
from elasticsearch import Elasticsearch
from concurrent.futures import ThreadPoolExecutor
import time
from llm_utils import summarize_content, extract_key_concepts, get_embedding

class AgenticSearchCrawler:
    def __init__(self):
        self.redis_queue = redis.Redis(host='localhost', port=6379, db=0)
        self.es = Elasticsearch(['http://localhost:9200'])
        self.visited_urls = set()
        self.max_depth = {
            'nrp': 4,
            'k8s': 2
        }
        self.base_urls = {
            'nrp': 'https://nrp.ai/documentation/',
            'k8s': 'https://kubernetes.io/docs/home/'
        }
        
    def process_page(self, task_data):
        url = task_data['url']
        depth = task_data['depth']
        source = task_data['source']
        
        if url in self.visited_urls or depth > self.max_depth[source]:
            return
            
        self.visited_urls.add(url)
        
        try:
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                return
                
            soup = BeautifulSoup(response.content, 'html.parser')
            content = self.extract_content(soup)
            links = self.extract_links(soup, url)
            
            # Use LLM to enhance content understanding
            print(f"Processing {url} with LLM...")
            summary = summarize_content(content)
            key_concepts = extract_key_concepts(content)
            embedding = get_embedding(content)
            
            # Index enhanced content
            self.index_content(url, content, source, summary, key_concepts, embedding)
            
            # Add links to queue
            for link in links:
                if link not in self.visited_urls:
                    self.redis_queue.lpush('crawl_queue', json.dumps({
                        'url': link,
                        'depth': depth + 1,
                        'source': source
                    }))
                    
        except Exception as e:
            print(f"Error processing {url}: {str(e)}")
    
    def index_content(self, url, content, source, summary, key_concepts, embedding):
        doc = {
            'url': url,
            'content': content,
            'source': source,
            'summary': summary,
            'key_concepts': key_concepts,
            'embedding': embedding,
            'timestamp': time.time()
        }
        
        self.es.index(index="documentation", body=doc)