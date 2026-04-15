import time
import sys

def main():
    for i in range(5):
        print(f"stdout step {i}", flush=True)
        print(f"stderr step {i}", file=sys.stderr, flush=True)
        time.sleep(5)

if __name__ == "__main__":
    main()
