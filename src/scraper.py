import requests
from bs4 import BeautifulSoup
import chromadb
from chromadb.config import Settings
import os
from dotenv import load_dotenv
import time

# 環境変数の読み込み
load_dotenv()

# データベースの保存ディレクトリを指定
DB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'fireworks_db')

# 保存ディレクトリが存在しない場合は作成
os.makedirs(DB_DIR, exist_ok=True)

# 花火に関するWebサイトのリスト（信頼性の高い日本語サイトのみ）
FIREWORKS_WEBSITES = [
    "https://www.oomagari-hanabi.com/index.html",
    "https://www.oomagari-hanabi.com/hanabi.html",
    "https://www.oomagari-hanabi.com/access.html",
    "https://www.oomagari-hanabi.com/ticket.html",

    "https://www.tsuchiura-kankou.jp/tomaru/",
    "https://www.tsuchiura-hanabi.jp/page/page000010.html",
    "https://www.tsuchiura-hanabi.jp/page/page000012.html",
    "https://www.tsuchiura-hanabi.jp/page/page000014.html",
    "https://www.tsuchiura-hanabi.jp/page/page000015.html",
    "https://www.tsuchiura-hanabi.jp/page/page000024.html"
    "https://nagaokamatsuri.com/"
    "https://nagaokamatsuri.com/faq/"
    "https://nagaokamatsuri.com/learn/"
    "https://nagaokamatsuri.com/history/"
    "https://nagaokamatsuri.com/launch/"
]

def scrape_website(url):
    try:
        # リクエストヘッダーを設定
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # タイムアウトを設定してリクエスト
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # エラーステータスコードの場合は例外を発生
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # メインコンテンツの取得（より具体的なセレクタを使用）
        main_content = soup.find('div', class_='article-body')
        if main_content:
            content = main_content.get_text(strip=True)
        else:
            content = soup.get_text(strip=True)
            
        return content
    except requests.exceptions.RequestException as e:
        print(f"Error scraping {url}: {str(e)}")
        return None
    except Exception as e:
        print(f"Unexpected error scraping {url}: {str(e)}")
        return None

def create_vector_db():
    # ChromaDBの初期化
    client = chromadb.PersistentClient(path=DB_DIR)
    
    # 既存のコレクションを削除（存在する場合）
    try:
        client.delete_collection(name="fireworks_information")
    except:
        pass
    
    # コレクションの作成
    collection = client.create_collection(name="fireworks_information")
    
    # 各Webサイトから情報を取得してベクトル化
    successful_scrapes = 0
    for i, url in enumerate(FIREWORKS_WEBSITES):
        print(f"Scraping {url}...")
        content = scrape_website(url)
        if content:
            # テキストをチャンクに分割（簡易的な実装）
            chunks = [content[i:i+1000] for i in range(0, len(content), 1000)]
            
            # 各チャンクをベクトルDBに追加
            for j, chunk in enumerate(chunks):
                collection.add(
                    documents=[chunk],
                    metadatas=[{"source": url}],
                    ids=[f"doc_{i}_{j}"]
                )
            successful_scrapes += 1
            print(f"Successfully scraped {url}")
        
        # サーバーに負荷をかけないように少し待機
        time.sleep(1)
    
    print(f"\nVector database created successfully at {DB_DIR}!")
    print(f"Successfully scraped {successful_scrapes} out of {len(FIREWORKS_WEBSITES)} websites")

if __name__ == "__main__":
    create_vector_db() 