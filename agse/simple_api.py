# simple_api.py
from flask import Flask, request, jsonify
from elasticsearch import Elasticsearch
from llm_utils import get_embedding, find_cross_reference, get_llm_response

app = Flask(__name__)
es = Elasticsearch(['http://localhost:9200'])

@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('q', '')
    if not query:
        return jsonify({'error': 'Query parameter "q" is required'}), 400
    
    try:
        # Get embedding for the query
        query_embedding = get_embedding(query)
        
        # Perform vector search
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
            "size": 10
        }
        
        response = es.search(index="documentation", body=script_query)
        hits = response['hits']['hits']
        
        results = []
        for hit in hits:
            source = hit['_source']
            results.append({
                'url': source['url'],
                'source': source['source'],
                'summary': source.get('summary', ''),
                'key_concepts': source.get('key_concepts', []),
                'score': hit['_score']
            })
        
        return jsonify({'results': results})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)