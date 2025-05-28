"""
Main application for the Cognitive Workflow System with DuckDuckGo Search
Fixed version - addresses task framing and runtime shutdown issues
"""

import asyncio
import os
import signal
import sys
from datetime import datetime
from dotenv import load_dotenv

from autogen_ext.models.openai import OpenAIChatCompletionClient

import ddg_search_module
import web_scraper_module
from cognitive_models import TaskNature
from cognitive_framer import MetaCognitiveFramer
from cognitive_workflow import CognitiveWorkflowSystem


# Global workflow system for cleanup
workflow_system = None


def signal_handler(signum, frame):
    """Handle interrupt signals gracefully"""
    print("\n🛑 Interrupt received, cleaning up...")
    if workflow_system:
        # Run cleanup in the event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Create a task for cleanup
                loop.create_task(workflow_system.cleanup_all_workflows())
            else:
                loop.run_until_complete(workflow_system.cleanup_all_workflows())
        except:
            pass
    
    print("👋 Goodbye!")
    sys.exit(0)


# Tool Functions with DuckDuckGo text search debugging only
async def search_web_information(query: str) -> str:
    """Search for information using DuckDuckGo text search only"""
    print(f"🔍 DDG DEBUG: search_web_information called with query: '{query}'")
    
    try:
        result = await ddg_search_module.search_web_information(
            query=query,
            max_results=8,
            include_news=False,  # Text search only
            timelimit="m",
            verbose=True
        )
        
        print(f"✅ DDG DEBUG: search_web_information completed")
        print(f"📊 DDG DEBUG: Result length: {len(result)} characters")
        print(f"📝 DDG DEBUG: Result preview: {result[:200]}...")
        return result
            
    except Exception as e:
        error_msg = f"Error performing web search: {str(e)}"
        print(f"❌ DDG DEBUG: {error_msg}")
        return error_msg


async def check_tools_availability():
    """Check if the required tools are available and properly configured."""
    print("🔧 Checking tool availability...")
    
    # Check DuckDuckGo search availability
    ddg_available = ddg_search_module.check_search_availability()
    print(f"🔍 DuckDuckGo Search Tool: {'✅ Available' if ddg_available else '❌ Missing Package'}")
    
    # Check web scraper health
    scraper_health = await web_scraper_module.check_scraper_health()
    scraper_available = scraper_health["node_js_available"] and scraper_health["script_exists"]
    print(f"🌐 Web Scraper Tool: {'✅ Available' if scraper_available else '❌ Missing Dependencies'}")
    
    if not ddg_available:
        print("⚠️  Warning: Install duckduckgo-search package: pip install duckduckgo-search")
    
    if not scraper_available:
        print("⚠️  Warning: Ensure Node.js is installed and scraper script exists")
    
    return ddg_available or scraper_available


async def main():
    global workflow_system
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    load_dotenv()
    TOGETHER_API_KEY = os.getenv('TOGETHER_API_KEY')

    print("🤖 Cognitive Workflow System with DuckDuckGo Search Engine")
    print("🧠 Dynamic Task Framing and Adaptive Execution")
    print("🔍 Powered by DuckDuckGo Search API")
    print(f"📅 Current Date: {datetime.now().strftime('%B %d, %Y')}")
    
    try:
        # Check tool availability
        tools_available = await check_tools_availability()
        if not tools_available:
            print("❌ No web tools available. Continuing with limited functionality...")
        
        # User input
        print("\n" + "="*60)
        print("📝 COGNITIVE TASK INPUT")
        print("="*60)
        
        user_request = input("Enter your request (can be simple question or complex meta-task): ").strip()
        
        if not user_request:
            user_request = "What are the latest developments in artificial intelligence in 2024 and how are they impacting different industries?"
            print(f"Using default query: {user_request}")
            
        # Create model client
        try:
            client = OpenAIChatCompletionClient(
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
                model_info={
                    "vision": False,
                    "function_calling": True,
                    "json_output": True,
                    "family": "unknown",
                    "structured_output": True
                },
                base_url="https://api.together.xyz/v1",
                api_key=TOGETHER_API_KEY
            )
            print("✅ Model client created successfully")
        except Exception as e:
            print(f"❌ Failed to create model client: {e}")
            return
        
        # Setup the cognitive workflow system
        workflow_system = CognitiveWorkflowSystem()
        
        # Create meta-cognitive framer
        framer = MetaCognitiveFramer(client)
        workflow_system.set_framer(framer)
        
        print(f"\n🧠 Analyzing task structure and intent...")
        
        # Frame the task using cognitive analysis
        task_frame = await workflow_system.frame_and_plan_task(user_request)
        
        # Create tool functions dictionary with text search only
        tool_functions = {
            "search_web_information": search_web_information,
        }
        
        # Create agents based on cognitive framing
        print(f"\n🤖 Creating specialized agents...")
        agents = workflow_system.create_cognitive_agents(client, task_frame, tool_functions)
        print(f"✅ Created {len(agents)} agents: {list(agents.keys())}")
        
        # Create adaptive workflow
        print(f"\n⚡ Building adaptive workflow topology...")
        workflow = workflow_system.create_adaptive_workflow(agents, task_frame)
        print(f"✅ Workflow created with {task_frame.execution_strategy.value} strategy")
        
        # Execute the cognitive workflow with proper cleanup
        print(f"\n🚀 Starting cognitive workflow execution...")
        
        final_result = await workflow_system.execute_cognitive_workflow(workflow, task_frame)
        
        # Display final summary
        print(f"\n{'🎯 FINAL RESULT':=^80}")
        print(final_result)
        print(f"{'='*80}")
        
        print(f"\n✅ Cognitive workflow completed successfully!")
        
    except KeyboardInterrupt:
        print("\n🛑 Application interrupted by user")
    except Exception as e:
        print(f"❌ Error during workflow execution: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Ensure cleanup happens
        if workflow_system:
            try:
                await workflow_system.cleanup_all_workflows()
                # Additional delay for complete cleanup
                await asyncio.sleep(0.2)
            except Exception as cleanup_error:
                print(f"⚠️ Final cleanup warning: {cleanup_error}")


def run_main():
    """Run the main function with proper event loop handling"""
    try:
        # For Python 3.7+
        if hasattr(asyncio, 'run'):
            asyncio.run(main())
        else:
            # Fallback for older Python versions
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(main())
            finally:
                # Clean up remaining tasks
                pending = asyncio.all_tasks(loop)
                if pending:
                    for task in pending:
                        task.cancel()
                    # Wait for cancellation to complete
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                loop.close()
    except KeyboardInterrupt:
        print("\n🛑 Application interrupted by user")
    except Exception as e:
        print(f"❌ Application failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_main()
