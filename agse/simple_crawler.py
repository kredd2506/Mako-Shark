# simple_crawler.py
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import json
from elasticsearch import Elasticsearch
from concurrent.futures import ThreadPoolExecutor
import time
from llm_utils import summarize_content, extract_key_concepts, get_embedding

class SimpleCrawler:
    def __init__(self):
        self.es = Elasticsearch(['http://localhost:9200'])
        
        # Create the index if it doesn't exist
        if not self.es.indices.exists(index="documentation"):
            print("Creating Elasticsearch index...")
            self.es.indices.create(index="documentation")
        
        self.visited_urls = set()
        self.max_depth = {
            'nrp': 3,
            'k8s': 2
        }
        self.base_urls = {
            'nrp': 'https://nrp.ai/documentation/',
            'k8s': 'https://kubernetes.io/docs/home/'
        }
        
    def start_crawling(self):
        # Start with base URLs
        with ThreadPoolExecutor(max_workers=4) as executor:
            for source, url in self.base_urls.items():
                executor.submit(self.process_page, url, 0, source)
    
    def process_page(self, url, depth, source):
        if url in self.visited_urls or depth > self.max_depth[source]:
            return
            
        self.visited_urls.add(url)
        print(f"Processing {url}...")
        
        try:
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                return
                
            soup = BeautifulSoup(response.content, 'html.parser')
            content = self.extract_content(soup)
            links = self.extract_links(soup, url)
            
            # Use LLM to enhance content understanding
            print(f"Enhancing {url} with LLM...")
            summary = summarize_content(content)
            key_concepts = extract_key_concepts(content)
            embedding = get_embedding(content)
            
            # Index content
            self.index_content(url, content, source, summary, key_concepts, embedding)
            
            # Process links
            for link in links:
                if link not in self.visited_urls:
                    self.process_page(link, depth + 1, source)
                    
        except Exception as e:
            print(f"Error processing {url}: {str(e)}")
    
    def extract_content(self, soup):
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
            
        # Extract text content
        content = ' '.join([p.get_text() for p in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])])
        return content.strip()
    
    def extract_links(self, soup, base_url):
        links = set()
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            if href.startswith('http'):
                # Only follow links from the same domain
                if urlparse(href).netloc == urlparse(base_url).netloc:
                    links.add(href)
            else:
                links.add(urljoin(base_url, href))
        return links
    
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

if __name__ == "__main__":
    crawler = SimpleCrawler()
    crawler.start_crawling()