import asyncio
import os
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse


class WebScraperClient:
    def __init__(
        self,
        env_file: str = "/home/jaykchen/projects/cat_scraper/.env",
        js_script: str = "/home/jaykchen/projects/cat_scraper/simple.js",
        timeout: float = 30.0,
        max_content_length: int = 50000,
        verbose: bool = True
    ):
        self.env_file = env_file
        self.js_script = js_script
        self.timeout = timeout
        self.max_content_length = max_content_length
        self.verbose = verbose
        
    async def scrape(self, url: str) -> str:
        return await scrape_webpage_with_js(
            url=url,
            env_file=self.env_file,
            js_script=self.js_script,
            timeout=self.timeout,
            max_content_length=self.max_content_length,
            verbose=self.verbose
        )
    


async def scrape_webpage_with_js(
    url: str,
    env_file: str = "/home/jaykchen/projects/cat_scraper/.env",
    js_script: str = "/home/jaykchen/projects/cat_scraper/simple.js",
    timeout: float = 30.0,
    max_content_length: int = 50000,
    verbose: bool = True
) -> str:
    try:
        if not url.startswith(('http://', 'https://')):
            return f"Error: Invalid URL format: {url}"
        
        try:
            parsed = urlparse(url)
            if not parsed.netloc:
                return f"Error: Invalid URL structure: {url}"
        except Exception:
            return f"Error: Could not parse URL: {url}"
        
        if verbose:
            print(f"🌐 Scraping webpage: {url}")
        
        if not os.path.isfile(js_script):
            return f"Error: Scraper script not found: {js_script}"
        
        proc = await asyncio.create_subprocess_exec(
            'node', f'--env-file={env_file}', js_script, url,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            return f"Error: Scraping timeout after {timeout} seconds for URL: {url}"
        
        if proc.returncode != 0:
            error_msg = stderr.decode('utf-8') if stderr else "Unknown error"
            return f"Error scraping {url}: {error_msg}"
        
        content = stdout.decode('utf-8')
        
        if not content.strip():
            return f"Warning: No content retrieved from {url}"
        
        if len(content) > max_content_length:
            content = content[:max_content_length] + f"\n\n... (truncated, original length was {len(content)} characters)"
        
        if verbose:
            print(f"✅ Successfully scraped {len(content)} characters from {url}")
        
        return content
        
    except FileNotFoundError:
        return "Error: Node.js not found. Please ensure Node.js is installed and 'node' is in your PATH."
    except Exception as e:
        return f"Error scraping webpage: {str(e)}"


async def check_scraper_health() -> Dict[str, Any]:
    health_info = {
        "node_js_available": False,
        "node_version": None,
        "script_exists": False
    }
    
    try:
        proc = await asyncio.create_subprocess_exec(
            'node', '--version',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode == 0:
            health_info["node_js_available"] = True
            health_info["node_version"] = stdout.decode().strip()
    except:
        pass
    
    health_info["script_exists"] = os.path.isfile("/home/jaykchen/projects/cat_scraper/simple.js")
    
    return health_info


if __name__ == "__main__":
    async def main():
        print("🌐 Web Scraper Module - Test")
        health = await check_scraper_health()
        print(f"Health: {health}")
        
        if health["node_js_available"] and health["script_exists"]:
            test_url = "https://www.example.com"
            result = await scrape_webpage_with_js(test_url)
            print(f"Result: {'Success' if not result.startswith('Error:') else 'Failed'}")
    
    asyncio.run(main())
