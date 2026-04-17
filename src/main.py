import time
import sys
from datetime import datetime


def ts():
    return datetime.now().strftime("%H:%M:%S")


def main():
    print(f"[{ts()}] 🚀 処理開始", flush=True)

    for i in range(1, 6):
        print(f"[{ts()}] 🟢 標準出力 | ステップ={i}", flush=True)
        print(f"[{ts()}] 🔴 標準エラー | ステップ={i}", file=sys.stderr, flush=True)
        time.sleep(10)

    print(f"[{ts()}] 🟢 標準出力 | 処理完了", flush=True)


if __name__ == "__main__":
    main()
