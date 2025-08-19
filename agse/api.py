# api.py - Updated API with LLM-powered search
from flask import Flask, request, jsonify
from search import DocumentationSearch

app = Flask(__name__)
search_engine = DocumentationSearch()

@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('q', '')
    if not query:
        return jsonify({'error': 'Query parameter "q" is required'}), 400
        
    results = search_engine.search(query)
    return jsonify({'results': results})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)