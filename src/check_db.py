import chromadb
from chromadb.config import Settings

def check_database():
    # ChromaDBの初期化
    client = chromadb.Client(Settings(
        persist_directory="data/fireworks_db"
    ))
    
    # コレクションの取得
    collection = client.get_collection(name="fireworks_information")
    
    # コレクション内のドキュメント数を取得
    count = collection.count()
    print(f"花火情報データベース内のドキュメント数: {count}")
    
    # 最初の10件のドキュメントを取得して表示
    results = collection.get(
        limit=10,
        include=['documents', 'metadatas']
    )
    
    print("\n最初の10件の花火情報:")
    print("-" * 50)
    
    for i, (doc, metadata) in enumerate(zip(results['documents'], results['metadatas']), 1):
        print(f"\nドキュメント {i}:")
        print(f"ソース: {metadata['source']}")
        print(f"内容: {doc[:200]}...")  # 最初の200文字のみ表示
        print("-" * 50)

def search_documents(query, n_results=3):
    # ChromaDBの初期化
    client = chromadb.Client(Settings(
        persist_directory="data/fireworks_db"
    ))
    
    # コレクションの取得
    collection = client.get_collection(name="fireworks_information")
    
    # クエリに基づいて類似ドキュメントを検索
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=['documents', 'metadatas', 'distances']
    )
    
    print(f"\n検索クエリ: {query}")
    print(f"検索結果数: {n_results}")
    print("-" * 50)
    
    for i, (doc, metadata, distance) in enumerate(zip(results['documents'][0], results['metadatas'][0], results['distances'][0]), 1):
        print(f"\n結果 {i}:")
        print(f"類似度スコア: {1 - distance:.4f}")  # 距離を類似度に変換
        print(f"ソース: {metadata['source']}")
        print(f"内容: {doc[:200]}...")  # 最初の200文字のみ表示
        print("-" * 50)

if __name__ == "__main__":
    while True:
        print("\n=== 花火情報データベース検索システム ===")
        print("1: データベースの概要を表示")
        print("2: 花火情報を検索")
        print("3: 終了")
        
        choice = input("\n選択してください (1-3): ")
        
        if choice == "1":
            check_database()
        elif choice == "2":
            query = input("検索したい花火に関する情報を入力してください: ")
            n_results = int(input("表示する結果の数 (1-10): "))
            search_documents(query, min(n_results, 10))
        elif choice == "3":
            print("システムを終了します。")
            break
        else:
            print("無効な選択です。1-3の数字を入力してください。") 