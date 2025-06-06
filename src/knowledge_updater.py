import os
import json
import chromadb
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import time
from urllib.parse import urljoin
import re
from googleapiclient.discovery import build
from dotenv import load_dotenv

class KnowledgeUpdater:
    def __init__(self):
        # 環境変数の読み込み
        load_dotenv()
        
        # Google Custom Search APIの設定
        self.api_key = os.getenv('GOOGLE_API_KEY')
        self.cse_id = os.getenv('GOOGLE_CSE_ID')
        
        if not self.api_key or not self.cse_id:
            raise ValueError("GOOGLE_API_KEYとGOOGLE_CSE_IDが設定されていません。.envファイルを確認してください。")
        
        # Custom Search APIサービスの初期化
        self.search_service = build('customsearch', 'v1', developerKey=self.api_key)
        
        # データベースの保存ディレクトリを指定
        self.DB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'fireworks_db')
        
        # ChromaDBの初期化
        self.client = chromadb.PersistentClient(path=self.DB_DIR)
        
        # コレクションの取得
        self.collection = self.client.get_collection(name="fireworks_information")
        self.unanswered_collection = self.client.get_collection(name="unanswered_questions")
        
        print(f"データベースディレクトリ: {self.DB_DIR}")
        print("コレクションの初期化が完了しました")
    
    def get_unanswered_questions(self):
        """
        未回答の質問を取得（知識更新済みの質問は除外）
        
        Returns:
            list: 未回答の質問のリスト
        """
        try:
            # メタデータでフィルタリング
            results = self.unanswered_collection.get(
                where={"is_updated": {"$ne": True}}  # 知識更新済みでない質問のみを取得
            )
            print(f"未回答の質問を取得しました: {len(results['ids'])}件")
            return results
        except Exception as e:
            print(f"未回答の質問の取得中にエラーが発生しました: {str(e)}")
            return None
    
    def search_google(self, keywords, num_results=2):
        """
        Google Custom Search APIを使用して検索を実行
        
        Args:
            keywords (list): 検索キーワード
            num_results (int): 取得する結果の数
        
        Returns:
            list: 検索結果のURLリスト
        """
        try:
            # キーワードを結合
            query = ' '.join(keywords)
            print(f"Google検索を実行: {query}")
            
            # Custom Search APIを使用して検索
            result = self.search_service.cse().list(
                q=query,
                cx=self.cse_id,
                num=num_results
            ).execute()
            
            # 検索結果のURLを抽出
            urls = []
            if 'items' in result:
                for item in result['items']:
                    url = item['link']
                    if not any(domain in url for domain in ['google.com', 'youtube.com']):
                        urls.append(url)
            
            print(f"検索結果のURL: {urls}")
            return urls
            
        except Exception as e:
            print(f"Google検索中にエラーが発生しました: {str(e)}")
            return []
    
    def scrape_webpage(self, url):
        """
        ウェブページをスクレイピング
        
        Args:
            url (str): スクレイピングするURL
        
        Returns:
            list: スクレイピングしたテキストのチャンクリスト
        """
        try:
            print(f"ウェブページをスクレイピング: {url}")
            
            # ヘッダーの設定
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            # ページの取得
            response = requests.get(url, headers=headers)
            response.raise_for_status()  # エラーチェック
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 不要な要素の削除
            for element in soup.find_all(['script', 'style', 'nav', 'footer', 'header']):
                element.decompose()
            
            # テキストの抽出
            text = soup.get_text()
            
            # テキストのクリーニング
            text = re.sub(r'\s+', ' ', text).strip()
            
            # テキストをチャンクに分割（1000文字ごと）
            chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
            
            print(f"スクレイピングしたテキストを{len(chunks)}個のチャンクに分割しました")
            return chunks
            
        except Exception as e:
            print(f"ウェブページのスクレイピング中にエラーが発生しました: {str(e)}")
            return []
    
    def add_knowledge(self, text, source, original_question=None):
        """
        知識をデータベースに追加
        
        Args:
            text (str): 追加するテキスト
            source (str): 情報源のURL
            original_question (str, optional): 元の質問
        """
        try:
            if not text:
                print("追加するテキストが空です")
                return
            
            # ドキュメントIDの生成
            doc_id = f"doc_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            # メタデータの作成
            metadata = {
                'source': source,
                'timestamp': datetime.now().isoformat(),
                'original_question': original_question if original_question else ''
            }
            
            print(f"知識を追加します: ID={doc_id}, ソース={source}")
            
            # コレクションに追加
            self.collection.add(
                documents=[text],
                metadatas=[metadata],
                ids=[doc_id]
            )
            
            # 追加の確認
            added_doc = self.collection.get(ids=[doc_id])
            if added_doc and len(added_doc['ids']) > 0:
                print(f"知識の追加に成功しました: {doc_id}")
            else:
                print(f"知識の追加に失敗しました: {doc_id}")
            
        except Exception as e:
            print(f"知識の追加中にエラーが発生しました: {str(e)}")
    
    def mark_question_as_updated(self, doc_id):
        """
        質問を知識更新済みとしてマーク
        
        Args:
            doc_id (str): 更新するドキュメントのID
        """
        try:
            # メタデータの更新
            self.unanswered_collection.update(
                ids=[doc_id],
                metadatas=[{
                    'is_updated': True,
                    'updated_at': datetime.now().isoformat()
                }]
            )
            print(f"質問を知識更新済みとしてマークしました: {doc_id}")
        except Exception as e:
            print(f"質問の更新状態の変更中にエラーが発生しました: {str(e)}")
    
    def update_knowledge(self):
        """
        未回答の質問の知識を更新
        """
        # 未回答の質問を取得
        results = self.get_unanswered_questions()
        if not results or not results['ids']:
            print("未回答の質問はありません。")
            return
        
        # 各質問について処理
        for doc, metadata, doc_id in zip(results['documents'], results['metadatas'], results['ids']):
            try:
                print(f"\n質問の処理を開始: {doc_id}")
                print(f"質問内容: {doc}")
                
                # キーワードを取得
                keywords = json.loads(metadata['keywords'])
                print(f"抽出されたキーワード: {keywords}")
                
                # Google検索を実行
                urls = self.search_google(keywords)
                
                if not urls:
                    print("検索結果が見つかりませんでした")
                    continue
                
                # 各URLについてスクレイピング
                for url in urls:
                    chunks = self.scrape_webpage(url)
                    if chunks:
                        # 各チャンクを知識として追加
                        for i, chunk in enumerate(chunks):
                            # チャンク番号をメタデータに追加
                            chunk_metadata = {
                                'source': url,
                                'timestamp': datetime.now().isoformat(),
                                'original_question': doc,
                                'chunk_index': i,
                                'total_chunks': len(chunks)
                            }
                            
                            # ドキュメントIDの生成（チャンク番号を含める）
                            chunk_id = f"doc_{datetime.now().strftime('%Y%m%d%H%M%S')}_{i}"
                            
                            # コレクションに追加
                            self.collection.add(
                                documents=[chunk],
                                metadatas=[chunk_metadata],
                                ids=[chunk_id]
                            )
                            
                            print(f"チャンク {i+1}/{len(chunks)} を追加しました")
                
                # 質問を知識更新済みとしてマーク
                self.mark_question_as_updated(doc_id)
                
                # サーバーに負荷をかけないように待機
                time.sleep(2)
            
            except Exception as e:
                print(f"質問の処理中にエラーが発生しました: {str(e)}")
                continue

if __name__ == "__main__":
    updater = KnowledgeUpdater()
    updater.update_knowledge() 