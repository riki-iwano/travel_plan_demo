import chromadb
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class TFIDFSearch:
    def __init__(self):
        # データベースの保存ディレクトリを指定
        self.DB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'fireworks_db')
        
        # ChromaDBの初期化
        self.client = chromadb.PersistentClient(path=self.DB_DIR)
        self.collection = self.client.get_collection(name="fireworks_information")
        
        # ドキュメントの取得とTF-IDFベクトライザーの初期化
        self._initialize_tfidf()
    
    def _initialize_tfidf(self):
        """TF-IDFベクトライザーを初期化し、ドキュメントをベクトル化"""
        # すべてのドキュメントを取得
        results = self.collection.get()
        self.documents = results['documents']
        self.metadatas = results['metadatas']
        self.ids = results['ids']
        
        # TF-IDFベクトライザーの初期化
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            token_pattern=r'(?u)\b\w+\b'
        )
        
        # ドキュメントをベクトル化
        self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)
    
    def search(self, query, n_results=3):
        """
        クエリに基づいて関連ドキュメントを検索
        
        Args:
            query (str): 検索クエリ
            n_results (int): 返す結果の数
        
        Returns:
            list: 検索結果のリスト（ドキュメントID、スコア、メタデータ、コンテンツを含む）
        """
        # クエリをベクトル化
        query_vector = self.vectorizer.transform([query])
        
        # コサイン類似度を計算
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # 上位n_results件のインデックスを取得
        top_indices = similarities.argsort()[-n_results:][::-1]
        
        # 結果を整形
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # 類似度が0より大きい場合のみ追加
                results.append({
                    'id': self.ids[idx],
                    'score': float(similarities[idx]),
                    'metadata': self.metadatas[idx],
                    'content': self.documents[idx]
                })
        
        return results

if __name__ == "__main__":
    # 使用例
    searcher = TFIDFSearch()
    results = searcher.search("握りずしの歴史")
    
    print("\n=== 検索結果 ===")
    for result in results:
        print(f"\nスコア: {result['score']:.4f}")
        print(f"ソース: {result['metadata']['source']}")
        print(f"内容: {result['content'][:200]}...") 