from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent.parent

REQUIRED_DIRECTORIES = [
    "data/raw",
    "data/processed"
]

def setup():
    for directory in REQUIRED_DIRECTORIES:
        path = ROOT_DIR / directory
        path.mkdir(parents=True, exist_ok=True)
        print(f"{directory} Created!!")

if __name__ == '__main__':
    setup()
    print("Setup completed")