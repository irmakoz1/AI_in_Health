import os
from anthropic import Anthropic
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
env_path = Path('.env')
if env_path.exists():
    load_dotenv()
    print(f"✅ Loaded .env from: {env_path.absolute()}")
else:
    print(f"❌ .env file not found at: {env_path.absolute()}")
    print("Create a .env file with: ANTHROPIC_API_KEY=your-key")
    exit(1)

client = Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY"),  # This is the default and can be omitted
)
page = client.models.list()
page = page.data[0]
print(page.id)