"""
DuckDuckGo search module to replace Anthropic search functionality
Provides comprehensive search capabilities including text, news, images, and videos
"""

import asyncio
import json
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
import traceback

try:
    from duckduckgo_search import DDGS
    from duckduckgo_search.exceptions import (
        DuckDuckGoSearchException,
        RatelimitException,
        TimeoutException,
    )
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False
    print("⚠️  duckduckgo-search not installed. Run: pip install duckduckgo-search")


class DuckDuckGoSearchClient:
    """
    Comprehensive DuckDuckGo search client with multiple search types
    """
    
    def __init__(
        self,
        proxy: Optional[str] = None,
        timeout: int = 15,
        verify_ssl: bool = True,
        max_retries: int = 3,
        verbose: bool = False
    ):
        self.proxy = proxy
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        self.max_retries = max_retries
        self.verbose = verbose
        
        if not DDGS_AVAILABLE:
            raise ImportError("duckduckgo_search package is required. Install with: pip install duckduckgo-search")
    
    def _create_ddgs_instance(self) -> DDGS:
        """Create a new DDGS instance with configured settings"""
        return DDGS(
            proxy=self.proxy,
            timeout=self.timeout,
            verify=self.verify_ssl
        )
    
    async def _execute_with_retry(self, search_func, *args, **kwargs) -> Any:
        """Execute search function with retry logic"""
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                if self.verbose and attempt > 0:
                    print(f"🔄 Retry attempt {attempt + 1}/{self.max_retries}")
                
                # Run the search function in a thread to avoid blocking
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, search_func, *args, **kwargs)
                return result
                
            except RatelimitException as e:
                last_exception = e
                if self.verbose:
                    print(f"⏳ Rate limit hit, waiting before retry {attempt + 1}")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                continue
                
            except TimeoutException as e:
                last_exception = e
                if self.verbose:
                    print(f"⏰ Timeout on attempt {attempt + 1}")
                continue
                
            except DuckDuckGoSearchException as e:
                last_exception = e
                if self.verbose:
                    print(f"🔍 Search error on attempt {attempt + 1}: {e}")
                continue
                
            except Exception as e:
                last_exception = e
                if self.verbose:
                    print(f"❌ Unexpected error on attempt {attempt + 1}: {e}")
                continue
        
        # All retries failed
        raise last_exception or Exception("All search attempts failed")
    
    async def search_text(
        self,
        query: str,
        region: str = "wt-wt",
        safesearch: str = "moderate",
        timelimit: Optional[str] = None,
        max_results: int = 10,
        backend: str = "auto"
    ) -> List[Dict[str, str]]:
        """
        Perform text search using DuckDuckGo
        
        Args:
            query: Search keywords
            region: Region code (wt-wt, us-en, uk-en, etc.)
            safesearch: on, moderate, off
            timelimit: d (day), w (week), m (month), y (year)
            max_results: Maximum number of results
            backend: auto, html, lite
            
        Returns:
            List of search results with title, href, body
        """
        if self.verbose:
            print(f"🔍 Searching text: '{query}' (max: {max_results})")
        
        def _search():
            ddgs = self._create_ddgs_instance()
            return list(ddgs.text(
                keywords=query,
                region=region,
                safesearch=safesearch,
                timelimit=timelimit,
                backend=backend,
                max_results=max_results
            ))
        
        try:
            results = await self._execute_with_retry(_search)
            if self.verbose:
                print(f"✅ Found {len(results)} text results")
            return results
        except Exception as e:
            if self.verbose:
                print(f"❌ Text search failed: {e}")
            return []
    
    async def search_news(
        self,
        query: str,
        region: str = "wt-wt",
        safesearch: str = "moderate",
        timelimit: Optional[str] = None,
        max_results: int = 10
    ) -> List[Dict[str, str]]:
        """
        Perform news search using DuckDuckGo
        
        Args:
            query: Search keywords
            region: Region code
            safesearch: on, moderate, off
            timelimit: d (day), w (week), m (month)
            max_results: Maximum number of results
            
        Returns:
            List of news results with date, title, body, url, image, source
        """
        if self.verbose:
            print(f"📰 Searching news: '{query}' (max: {max_results})")
        
        def _search():
            ddgs = self._create_ddgs_instance()
            return list(ddgs.news(
                keywords=query,
                region=region,
                safesearch=safesearch,
                timelimit=timelimit,
                max_results=max_results
            ))
        
        try:
            results = await self._execute_with_retry(_search)
            if self.verbose:
                print(f"✅ Found {len(results)} news results")
            return results
        except Exception as e:
            if self.verbose:
                print(f"❌ News search failed: {e}")
            return []
    
    async def search_images(
        self,
        query: str,
        region: str = "wt-wt",
        safesearch: str = "moderate",
        timelimit: Optional[str] = None,
        size: Optional[str] = None,
        color: Optional[str] = None,
        type_image: Optional[str] = None,
        layout: Optional[str] = None,
        license_image: Optional[str] = None,
        max_results: int = 10
    ) -> List[Dict[str, str]]:
        """
        Perform image search using DuckDuckGo
        
        Args:
            query: Search keywords
            region: Region code
            safesearch: on, moderate, off
            timelimit: Day, Week, Month, Year
            size: Small, Medium, Large, Wallpaper
            color: color, Monochrome, Red, Orange, Yellow, Green, Blue, Purple, Pink, Brown, Black, Gray, Teal, White
            type_image: photo, clipart, gif, transparent, line
            layout: Square, Tall, Wide
            license_image: any, Public, Share, ShareCommercially, Modify, ModifyCommercially
            max_results: Maximum number of results
            
        Returns:
            List of image results
        """
        if self.verbose:
            print(f"🖼️  Searching images: '{query}' (max: {max_results})")
        
        def _search():
            ddgs = self._create_ddgs_instance()
            return list(ddgs.images(
                keywords=query,
                region=region,
                safesearch=safesearch,
                timelimit=timelimit,
                size=size,
                color=color,
                type_image=type_image,
                layout=layout,
                license_image=license_image,
                max_results=max_results
            ))
        
        try:
            results = await self._execute_with_retry(_search)
            if self.verbose:
                print(f"✅ Found {len(results)} image results")
            return results
        except Exception as e:
            if self.verbose:
                print(f"❌ Image search failed: {e}")
            return []
    
    async def search_videos(
        self,
        query: str,
        region: str = "wt-wt",
        safesearch: str = "moderate",
        timelimit: Optional[str] = None,
        resolution: Optional[str] = None,
        duration: Optional[str] = None,
        license_videos: Optional[str] = None,
        max_results: int = 10
    ) -> List[Dict[str, str]]:
        """
        Perform video search using DuckDuckGo
        
        Args:
            query: Search keywords
            region: Region code
            safesearch: on, moderate, off
            timelimit: d (day), w (week), m (month)
            resolution: high, standard
            duration: short, medium, long
            license_videos: creativeCommon, youtube
            max_results: Maximum number of results
            
        Returns:
            List of video results
        """
        if self.verbose:
            print(f"🎥 Searching videos: '{query}' (max: {max_results})")
        
        def _search():
            ddgs = self._create_ddgs_instance()
            return list(ddgs.videos(
                keywords=query,
                region=region,
                safesearch=safesearch,
                timelimit=timelimit,
                resolution=resolution,
                duration=duration,
                license_videos=license_videos,
                max_results=max_results
            ))
        
        try:
            results = await self._execute_with_retry(_search)
            if self.verbose:
                print(f"✅ Found {len(results)} video results")
            return results
        except Exception as e:
            if self.verbose:
                print(f"❌ Video search failed: {e}")
            return []
    
    async def comprehensive_search(
        self,
        query: str,
        include_text: bool = True,
        include_news: bool = True,
        max_results_per_type: int = 5,
        **kwargs
    ) -> Dict[str, List[Dict]]:
        """
        Perform comprehensive search across multiple content types
        
        Args:
            query: Search keywords
            include_text: Include text search results
            include_news: Include news search results
            max_results_per_type: Maximum results per search type
            **kwargs: Additional arguments passed to individual search methods
            
        Returns:
            Dictionary with results by search type
        """
        if self.verbose:
            print(f"🔍 Comprehensive search: '{query}'")
        
        results = {}
        search_tasks = []
        
        if include_text:
            search_tasks.append(("text", self.search_text(query, max_results=max_results_per_type, **kwargs)))
        
        if include_news:
            search_tasks.append(("news", self.search_news(query, max_results=max_results_per_type, **kwargs)))
                
        # Execute all searches concurrently
        search_results = await asyncio.gather(*[task[1] for task in search_tasks], return_exceptions=True)
        
        for i, (search_type, _) in enumerate(search_tasks):
            result = search_results[i]
            if isinstance(result, Exception):
                if self.verbose:
                    print(f"❌ {search_type} search failed: {result}")
                results[search_type] = []
            else:
                results[search_type] = result
        
        return results
    
    def format_search_results(
        self,
        results: Union[List[Dict], Dict[str, List[Dict]]],
        max_length: int = 200
    ) -> str:
        """
        Format search results into a readable text format
        
        Args:
            results: Search results from any search method
            max_length: Maximum length for result descriptions
            
        Returns:
            Formatted string representation of results
        """
        if isinstance(results, dict):
            # Comprehensive results
            formatted_parts = []
            for search_type, type_results in results.items():
                if type_results:
                    formatted_parts.append(f"\n{'='*20} {search_type.upper()} RESULTS {'='*20}")
                    for i, result in enumerate(type_results, 1):
                        formatted_parts.append(self._format_single_result(result, i, max_length))
            return "\n".join(formatted_parts)
        else:
            # Single type results
            formatted_parts = []
            for i, result in enumerate(results, 1):
                formatted_parts.append(self._format_single_result(result, i, max_length))
            return "\n".join(formatted_parts)
    
    def _format_single_result(self, result: Dict, index: int, max_length: int) -> str:
        """Format a single search result"""
        title = result.get('title', 'No title')
        url = result.get('href') or result.get('url', 'No URL')
        
        # Get description from various possible fields
        description = (
            result.get('body') or 
            result.get('description') or 
            result.get('content', '')
        )
        
        if len(description) > max_length:
            description = description[:max_length] + "..."
        
        # Add date for news results
        date_info = ""
        if 'date' in result:
            try:
                date_obj = datetime.fromisoformat(result['date'].replace('Z', '+00:00'))
                date_info = f" [{date_obj.strftime('%Y-%m-%d')}]"
            except:
                date_info = f" [{result['date']}]"
        
        return f"{index}. {title}{date_info}\n   URL: {url}\n   {description}\n"


# Convenience functions for backward compatibility and ease of use
async def search_web_information(
    query: str,
    max_results: int = 10,
    include_news: bool = True,
    timelimit: Optional[str] = None,
    verbose: bool = False
) -> str:
    """
    Main search function to replace anthropic_search_module.search_and_extract_text
    
    Args:
        query: Search query
        max_results: Maximum number of results per type
        include_news: Whether to include news results
        timelimit: Time limit for search (d, w, m, y)
        verbose: Whether to print verbose output
        
    Returns:
        Formatted search results as string
    """
    try:
        client = DuckDuckGoSearchClient(verbose=verbose, timeout=20)
        
        if include_news:
            results = await client.comprehensive_search(
                query,
                include_text=True,
                include_news=True,
                max_results_per_type=max_results,
                timelimit=timelimit
            )
        else:
            text_results = await client.search_text(
                query,
                max_results=max_results,
                timelimit=timelimit
            )
            results = {"text": text_results}
        
        formatted_results = client.format_search_results(results)
        
        if formatted_results.strip():
            return f"Search results for '{query}':\n{formatted_results}"
        else:
            return f"No results found for query: {query}"
            
    except Exception as e:
        error_msg = f"Error performing web search: {str(e)}"
        if verbose:
            print(f"❌ {error_msg}")
            traceback.print_exc()
        return error_msg



async def verify_information_with_search(
    claim: str,
    max_results: int = 5,
    verbose: bool = False
) -> str:
    """
    Verify information by searching for supporting evidence
    
    Args:
        claim: Information to verify
        max_results: Maximum number of results
        verbose: Whether to print verbose output
        
    Returns:
        Formatted verification results as string
    """
    try:
        # Create verification-focused queries
        verification_queries = [
            f"verify {claim}",
            f"fact check {claim}",
            f"{claim} evidence",
            f"{claim} research study"
        ]
        
        client = DuckDuckGoSearchClient(verbose=verbose)
        all_results = []
        
        # Search with multiple verification-focused queries
        for query in verification_queries[:2]:  # Limit to first 2 queries
            results = await client.search_text(
                query,
                max_results=max_results // 2,
                timelimit="y"  # Focus on more recent information
            )
            all_results.extend(results)
        
        # Also search recent news for any updates
        news_results = await client.search_news(
            claim,
            max_results=max_results // 2,
            timelimit="m"
        )
        
        if all_results or news_results:
            formatted_text = ""
            if all_results:
                formatted_text += client.format_search_results(all_results)
            if news_results:
                formatted_text += "\n" + "="*20 + " RECENT NEWS " + "="*20 + "\n"
                formatted_text += client.format_search_results(news_results)
            
            return f"Verification search for '{claim}':\n{formatted_text}"
        else:
            return f"Could not find verification information for: {claim}"
            
    except Exception as e:
        error_msg = f"Error during verification search: {str(e)}"
        if verbose:
            print(f"❌ {error_msg}")
        return error_msg


def check_search_availability() -> bool:
    """Check if DuckDuckGo search is available"""
    return DDGS_AVAILABLE


if __name__ == "__main__":
    async def main():
        print("🔍 DuckDuckGo Search Module - Test Suite")
        print("="*50)
        
        # Check availability
        if not check_search_availability():
            print("❌ DuckDuckGo search package not available")
            print("📦 Install with: pip install duckduckgo-search")
            return
        

        client = DuckDuckGoSearchClient(verbose=True)
        
        # Test comprehensive search
        results = await client.comprehensive_search(
            "beef price in Ontario",
            include_text=True,
            include_news=True,
            max_results_per_type=2
        )
        
        print("\n📊 Comprehensive search results:")
        formatted = client.format_search_results(results)
        print(formatted[:500] + "..." if len(formatted) > 500 else formatted)
        
        print("\n✅ Test suite completed")
    
    asyncio.run(main())
