import anthropic
from dotenv import load_dotenv
from typing import Optional
import os

load_dotenv()


class AnthropicSearchClient:
    def __init__(
        self, 
        api_key: Optional[str] = None,
        model: str = "claude-3-5-haiku-20241022",
        max_tokens: int = 4000,
        verbose: bool = False
    ):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.verbose = verbose
    
    def search_and_extract_text(self, query: str) -> Optional[str]:
        messages = [{"role": "user", "content": f"{query}"}]
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=messages,
                tools=[{"type": "web_search_20250305", "name": "web_search"}]
            )
            
            if self.verbose:
                print("=" * 80)
                for content_block in response.content:
                    if content_block.type == "text":
                        print(content_block.text)
                    elif content_block.type == "tool_use":
                        print(f"✅ Used tool: {content_block.name}")
            
            text_content = []
            for content_block in response.content:
                if content_block.type == "text":
                    text_content.append(content_block.text)
            
            return "\n".join(text_content) if text_content else None
            
        except Exception as e:
            if self.verbose:
                print(f"❌ Error: {e}")
            return None


def search_and_extract_text(
    query: str,
    verbose: bool = False,
    model: str = "claude-3-5-haiku-20241022",
    max_tokens: int = 4000
) -> Optional[str]:
    client = AnthropicSearchClient(
        model=model,
        max_tokens=max_tokens,
        verbose=verbose
    )
    return client.search_and_extract_text(query)


def check_api_key() -> bool:
    api_key = os.getenv('ANTHROPIC_API_KEY')
    return api_key is not None and len(api_key.strip()) > 0


if __name__ == "__main__":
    if not check_api_key():
        print("❌ Error: ANTHROPIC_API_KEY not found")
        exit(1)
    
    query = "search for Kitchener's weather today"
    result = search_and_extract_text(query, verbose=True)
    
    if result:
        print("✅ Search completed successfully")
    else:
        print("❌ Search failed")
