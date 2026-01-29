"""
One-off script: bật extension pgvector trên Render Postgres.
Chạy từ máy local (không cần psql).

Cách dùng:
  1. Trên Render Dashboard → Postgres → Info → copy "External Database URL"
  2. Trong .env thêm 1 dòng (chỉ để chạy script này, xóa sau nếu muốn):
     RENDER_DATABASE_URL=postgresql://user:pass@host/dbname?sslmode=require
  3. Chạy: python enable_pgvector.py
"""
import os
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Ưu tiên RENDER_DATABASE_URL (External URL từ Render), không thì DATABASE_URL
url = os.environ.get("RENDER_DATABASE_URL") or os.environ.get("DATABASE_URL")
if not url or "127.0.0.1" in url or "localhost" in url:
    print("Set RENDER_DATABASE_URL or DATABASE_URL = Render External Database URL in .env")
    sys.exit(1)

try:
    import psycopg2
except ImportError:
    print("Install: pip install psycopg2-binary")
    sys.exit(1)

def main():
    print("Connecting to Render Postgres...")
    conn = psycopg2.connect(url)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    print("Done: CREATE EXTENSION IF NOT EXISTS vector;")
    cur.close()
    conn.close()
    print("OK. Redeploy Web Service on Render.")

if __name__ == "__main__":
    main()
