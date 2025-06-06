import os
import chromadb
import pandas as pd
from datetime import datetime

class DatabaseExporter:
    def __init__(self):
        # データベースの保存ディレクトリを指定
        self.DB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'fireworks_db')
        
        # ChromaDBの初期化
        self.client = chromadb.PersistentClient(path=self.DB_DIR)
        
        # コレクションの取得
        self.collection = self.client.get_collection(name="fireworks_information")
        self.unanswered_collection = self.client.get_collection(name="unanswered_questions")
    
    def export_to_csv(self):
        """
        データベースの内容をCSVファイルに出力
        """
        try:
            # メインコレクションのデータを取得
            main_results = self.collection.get()
            
            # 未回答の質問コレクションのデータを取得
            unanswered_results = self.unanswered_collection.get()
            
            # メインコレクションのデータをDataFrameに変換
            main_data = []
            for doc, metadata, doc_id in zip(main_results['documents'], main_results['metadatas'], main_results['ids']):
                main_data.append({
                    'id': doc_id,
                    'content': doc,
                    'source': metadata.get('source', ''),
                    'timestamp': metadata.get('timestamp', '')
                })
            
            main_df = pd.DataFrame(main_data)
            
            # 未回答の質問のデータをDataFrameに変換
            unanswered_data = []
            for doc, metadata, doc_id in zip(unanswered_results['documents'], unanswered_results['metadatas'], unanswered_results['ids']):
                unanswered_data.append({
                    'id': doc_id,
                    'question': doc,
                    'keywords': metadata.get('keywords', ''),
                    'timestamp': metadata.get('timestamp', '')
                })
            
            unanswered_df = pd.DataFrame(unanswered_data)
            
            # 出力ディレクトリの作成
            output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'exports')
            os.makedirs(output_dir, exist_ok=True)
            
            # タイムスタンプ付きのファイル名を生成
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            main_csv_path = os.path.join(output_dir, f'fireworks_information_{timestamp}.csv')
            unanswered_csv_path = os.path.join(output_dir, f'unanswered_questions_{timestamp}.csv')
            
            # CSVファイルに出力
            main_df.to_csv(main_csv_path, index=False, encoding='utf-8-sig')
            unanswered_df.to_csv(unanswered_csv_path, index=False, encoding='utf-8-sig')
            
            print(f"メインコレクションのデータを出力しました: {main_csv_path}")
            print(f"未回答の質問のデータを出力しました: {unanswered_csv_path}")
            
            # データの概要を表示
            print("\n=== データの概要 ===")
            print(f"メインコレクションのドキュメント数: {len(main_df)}")
            print(f"未回答の質問の数: {len(unanswered_df)}")
            
        except Exception as e:
            print(f"データの出力中にエラーが発生しました: {str(e)}")

if __name__ == "__main__":
    exporter = DatabaseExporter()
    exporter.export_to_csv() 