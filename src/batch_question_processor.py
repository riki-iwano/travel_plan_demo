import pandas as pd
import json
from datetime import datetime
from rag_system import FireworksRAGSystem
import os
import chromadb
from chromadb.config import Settings

class BatchQuestionProcessor:
    def __init__(self):
        self.rag_system = FireworksRAGSystem()
        self.DB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'fireworks_db')
        
        # ChromaDBの初期化
        self.client = chromadb.PersistentClient(path=self.DB_DIR)
        self.unanswered_collection = self.client.get_or_create_collection(name="unanswered_questions")
        
        # 結果保存用のディレクトリ
        self.results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'question_results')
        os.makedirs(self.results_dir, exist_ok=True)
    
    def process_questions(self, csv_file):
        """CSVファイルから質問を読み込み、RAGシステムで処理"""
        try:
            # CSVファイルの読み込み
            df = pd.read_csv(csv_file)
            total_questions = len(df)
            answered_questions = 0
            unanswered_questions = 0
            
            # 結果を格納するリスト
            results = []
            
            # 各質問を処理
            for index, row in df.iterrows():
                print(f"\n処理中の質問 {index + 1}/{total_questions}")
                print(f"質問者: {row['氏名']} ({row['年齢']}歳)")
                print(f"質問: {row['質問']}")
                
                # キーワード抽出
                keywords = self.rag_system.extract_keywords(row['質問'])
                print(f"抽出されたキーワード: {keywords}")
                
                # RAGシステムで質問を処理
                response = self.rag_system.process_query(row['質問'])
                
                # 結果を保存
                result = {
                    'persona': {
                        'nationality': row['国籍'],
                        'name': row['氏名'],
                        'age': row['年齢'],
                        'income': row['年収'],
                        'interest_japan': row['日本への関心度'],
                        'interest_fireworks': row['花火への関心度'],
                        'concerns': row['質問にあたって気になっていること']
                    },
                    'question': row['質問'],
                    'keywords': keywords,
                    'response': response['response'],
                    'relevant_docs': response['relevant_docs'],
                    'is_answered': len(response['relevant_docs']) > 0,
                    'timestamp': datetime.now().isoformat()
                }
                
                # 未回答の場合、unanswered_questionsコレクションに追加
                if not result['is_answered']:
                    self.unanswered_collection.add(
                        documents=[row['質問']],
                        metadatas=[{
                            'persona': json.dumps(result['persona']),
                            'keywords': json.dumps(keywords),
                            'timestamp': datetime.now().isoformat(),
                            'is_updated': False
                        }],
                        ids=[f"unanswered_{datetime.now().timestamp()}"]
                    )
                    unanswered_questions += 1
                    print(f"未回答の質問を保存しました: {row['質問']}")
                else:
                    answered_questions += 1
                
                print(f"回答: {result['response']}")
                print(f"関連ドキュメント数: {len(result['relevant_docs'])}")
                print("-" * 80)
                
                results.append(result)
            
            # 結果をJSONファイルに保存
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = os.path.join(self.results_dir, f'question_results_{timestamp}.json')
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            print(f"\n処理完了！結果は {output_file} に保存されました。")
            print("\n=== 統計情報 ===")
            print(f"総質問数: {total_questions}")
            print(f"回答可能な質問数: {answered_questions}")
            print(f"未回答の質問数: {unanswered_questions}")
            print(f"回答率: {(answered_questions/total_questions)*100:.1f}%")
            
        except Exception as e:
            print(f"エラーが発生しました: {str(e)}")

def main():
    processor = BatchQuestionProcessor()
    csv_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                           'data', 'persona_question', 'questions.csv')
    processor.process_questions(csv_file)

if __name__ == "__main__":
    main() 