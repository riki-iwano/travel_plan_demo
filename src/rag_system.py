import os
from dotenv import load_dotenv
import google.generativeai as genai
from tfidf_search import TFIDFSearch
import chromadb
import json
from datetime import datetime

# 環境変数の読み込み
load_dotenv()

class FireworksRAGSystem:
    def __init__(self):
        # Google APIキーの設定
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEYが設定されていません。.envファイルを確認してください。")
        
        # Gemini APIの初期化
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-pro')
        
        # データベースの保存ディレクトリを指定
        self.DB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'fireworks_db')
        
        # ChromaDBの初期化
        self.client = chromadb.PersistentClient(path=self.DB_DIR)
        
        # 既存のコレクションを取得または作成
        try:
            self.collection = self.client.get_collection(name="fireworks_information")
        except:
            self.collection = self.client.create_collection(name="fireworks_information")
        
        # 未回答の質問を保存するコレクションを取得または作成
        try:
            self.unanswered_collection = self.client.get_collection(name="unanswered_questions")
        except:
            self.unanswered_collection = self.client.create_collection(name="unanswered_questions")
        
        # TF-IDF検索システムの初期化
        self.search_system = TFIDFSearch()
        
        # テスト応答の生成
        try:
            response = self.model.generate_content("こんにちは")
            print("Gemini APIの初期化に成功しました")
        except Exception as e:
            print(f"Gemini APIの初期化中にエラーが発生しました: {str(e)}")
            raise
    
    def extract_keywords(self, query):
        """
        質問文から重要なキーワードを抽出
        
        Args:
            query (str): ユーザーの質問文
        
        Returns:
            list: 抽出されたキーワードのリスト
        """
        try:
            prompt = f"""
            以下の質問文から、検索に使用する重要なキーワードを抽出してください。
            キーワードは日本語で、最大3個までカンマ区切りで出力してください。
            質問文の意図を理解し、関連する重要な単語を抽出してください。
            
            質問文: {query}
            
            例：
            入力: 「花火大会の歴史について教えてください」
            出力: 花火大会,歴史
            
            入力: 「花火の種類と特徴を説明してください」
            出力: 花火,種類,特徴
            """
            
            response = self.model.generate_content(prompt)
            keywords = [kw.strip() for kw in response.text.split(',')]
            return keywords
        except Exception as e:
            print(f"キーワード抽出中にエラーが発生しました: {str(e)}")
            return query.split()  # エラー時は単純な分割を使用
    
    def get_relevant_documents(self, query, n_results=3):
        """
        クエリに関連するドキュメントを検索
        
        Args:
            query (str): 検索クエリ
            n_results (int): 返す結果の数
        
        Returns:
            list: 関連ドキュメントのリスト
        """
        try:
            # キーワードを抽出
            keywords = self.extract_keywords(query)
            print(f"抽出されたキーワード: {keywords}")
            
            # キーワードを結合して検索クエリを作成
            search_query = ' '.join(keywords)
            results = self.search_system.search(search_query, n_results)
            return results
        except Exception as e:
            print(f"ドキュメント検索中にエラーが発生しました: {str(e)}")
            return []
    
    def save_unanswered_question(self, query, keywords):
        """
        未回答の質問を保存
        
        Args:
            query (str): ユーザーの質問
            keywords (list): 抽出されたキーワード
        """
        try:
            # メタデータの作成
            metadata = {
                'query': query,
                'keywords': json.dumps(keywords, ensure_ascii=False),
                'timestamp': datetime.now().isoformat()
            }
            
            # ドキュメントIDの生成
            doc_id = f"unanswered_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            # コレクションに追加
            self.unanswered_collection.add(
                documents=[query],
                metadatas=[metadata],
                ids=[doc_id]
            )
            
            print(f"未回答の質問を保存しました: {query}")
        except Exception as e:
            print(f"未回答の質問の保存中にエラーが発生しました: {str(e)}")
    
    def generate_response(self, query, relevant_docs):
        """
        クエリと関連ドキュメントに基づいて応答を生成
        
        Args:
            query (str): ユーザーの質問
            relevant_docs (list): 関連ドキュメントのリスト
        
        Returns:
            str: 生成された応答
        """
        try:
            # 関連ドキュメントがない場合
            if not relevant_docs:
                return "ごめんね、わかりません😭"
            
            # プロンプトの作成
            context = "\n\n".join([doc['content'] for doc in relevant_docs])
            prompt = f"""
            以下の情報を参考に、質問に答えてください。
            
            参考情報:
            {context}
            
            質問: {query}
            
            回答は日本語で、簡潔かつ正確に提供してください。
            参考情報に基づいて回答し、情報が不足している場合はその旨を明記してください。
            花火に関する専門的な情報を提供する際は、安全性や法的な制約についても言及してください。
            """
            
            # 応答の生成
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"応答生成中にエラーが発生しました: {str(e)}")
            return "申し訳ありません。回答の生成中にエラーが発生しました。"
    
    def process_query(self, query):
        """
        ユーザーのクエリを処理し、応答を生成
        
        Args:
            query (str): ユーザーの質問
        
        Returns:
            dict: 処理結果（応答、関連ドキュメント、キーワードを含む）
        """
        try:
            # キーワードの抽出
            keywords = self.extract_keywords(query)
            
            # 関連ドキュメントの取得
            relevant_docs = self.get_relevant_documents(query)
            
            # 関連ドキュメントがない場合、質問を保存
            if not relevant_docs:
                self.save_unanswered_question(query, keywords)
            
            # 応答の生成
            response = self.generate_response(query, relevant_docs)
            
            return {
                'response': response,
                'relevant_docs': relevant_docs,
                'keywords': keywords
            }
        except Exception as e:
            print(f"クエリ処理中にエラーが発生しました: {str(e)}")
            return {
                'response': "申し訳ありません。処理中にエラーが発生しました。",
                'relevant_docs': [],
                'keywords': []
            } 