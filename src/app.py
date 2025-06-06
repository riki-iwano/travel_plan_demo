from flask import Flask, request, jsonify, render_template
from rag_system import FireworksRAGSystem
import os

# テンプレートディレクトリのパスを設定
template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'templates'))
app = Flask(__name__, template_folder=template_dir)
rag_system = FireworksRAGSystem()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    try:
        data = request.get_json()
        query_text = data.get('query', '').strip()
        
        if not query_text:
            return jsonify({
                'error': 'クエリが空です。'
            }), 400
        
        # クエリの処理
        result = rag_system.process_query(query_text)
        return jsonify(result)
    
    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")
        return jsonify({
            'error': '処理中にエラーが発生しました。'
        }), 500

@app.route('/search', methods=['POST'])
def search():
    try:
        data = request.get_json()
        query_text = data.get('query', '').strip()
        count = data.get('count', 3)
        
        if not query_text:
            return jsonify({
                'error': '検索キーワードが空です。'
            }), 400
        
        # TF-IDF検索の実行
        results = rag_system.search_system.search(query_text, n_results=count)
        
        # 結果の整形
        formatted_results = []
        for result in results:
            formatted_results.append({
                'source': result['metadata']['source'],
                'score': f"{(result['score'] * 100):.2f}%",
                'content': result['content']
            })
        
        return jsonify({'results': formatted_results})
    
    except Exception as e:
        print(f"検索中にエラーが発生しました: {str(e)}")
        return jsonify({
            'error': '検索中にエラーが発生しました。'
        }), 500

if __name__ == '__main__':
    app.run(debug=True) 