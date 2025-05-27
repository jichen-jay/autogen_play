import asyncio
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv
import os

from autogen_agentchat.agents import AssistantAgent, MessageFilterAgent, MessageFilterConfig, PerSourceFilter
from autogen_agentchat.teams import DiGraphBuilder, GraphFlow
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient

import anthropic_search_module
import web_scraper_module


class CognitiveDimension(Enum):
    KNOWLEDGE_ACQUISITION = "knowledge_acquisition"
    ANALYSIS = "analysis"
    SYNTHESIS = "synthesis"
    METACOGNITION = "metacognition"
    COMMUNICATION = "communication"

class ProblemSolvingStage(Enum):
    UNDERSTAND = "understand"
    DEVISE = "devise"
    CARRY_OUT = "carry_out"
    LOOK_BACK = "look_back"

@dataclass
class RoleFramework:
    name: str
    cognitive_dimensions: List[CognitiveDimension]
    problem_solving_stages: List[ProblemSolvingStage]


ROLE_FRAMEWORKS = {
    "collector": RoleFramework(
        name="Information Collector",
        cognitive_dimensions=[CognitiveDimension.KNOWLEDGE_ACQUISITION],
        problem_solving_stages=[ProblemSolvingStage.UNDERSTAND],
    ),
    
    "analyzer": RoleFramework(
        name="Pattern Analyzer", 
        cognitive_dimensions=[CognitiveDimension.ANALYSIS],
        problem_solving_stages=[ProblemSolvingStage.UNDERSTAND, ProblemSolvingStage.DEVISE],
    ),
    
    "synthesizer": RoleFramework(
        name="Knowledge Synthesizer",
        cognitive_dimensions=[CognitiveDimension.SYNTHESIS],
        problem_solving_stages=[ProblemSolvingStage.DEVISE, ProblemSolvingStage.CARRY_OUT],
    ),
    
    "validator": RoleFramework(
        name="Quality Validator",
        cognitive_dimensions=[CognitiveDimension.METACOGNITION],
        problem_solving_stages=[ProblemSolvingStage.LOOK_BACK],
    ),
    
    "communicator": RoleFramework(
        name="Final Report Generator",
        cognitive_dimensions=[CognitiveDimension.COMMUNICATION],
        problem_solving_stages=[ProblemSolvingStage.CARRY_OUT],
    )
}


async def search_web_information(query: str) -> str:
    try:
        result = anthropic_search_module.search_and_extract_text(
            query=query,
            verbose=True,
            model="claude-3-5-haiku-20241022",
            max_tokens=4000
        )
        
        if result:
            return f"Web search results for '{query}':\n\n{result}"
        else:
            return f"No results found for query: {query}"
            
    except Exception as e:
        return f"Error performing web search: {str(e)}"


async def scrape_webpage_content(url: str) -> str:
    try:
        result = await web_scraper_module.scrape_webpage_with_js(
            url=url,
            timeout=30.0,
            max_content_length=50000,
            verbose=True
        )
        
        if result and not result.startswith("Error:"):
            return f"Content from {url}:\n\n{result}"
        else:
            return f"Failed to scrape {url}: {result}"
            
    except Exception as e:
        return f"Error scraping webpage: {str(e)}"


async def verify_information_with_search(claim: str) -> str:
    try:
        verification_query = f"verify information: {claim}"
        
        result = anthropic_search_module.search_and_extract_text(
            query=verification_query,
            verbose=True,
            model="claude-3-5-haiku-20241022",
            max_tokens=3000
        )
        
        if result:
            return f"Verification search for '{claim}':\n\n{result}"
        else:
            return f"Could not verify information about: {claim}"
            
    except Exception as e:
        return f"Error during verification search: {str(e)}"


class WorkflowVisualizer:
    def __init__(self):
        self.step_count = 0
        self.stage_history = []
        
    def print_stage_header(self, stage_name: str, agent_name: str, description: str = ""):
        self.step_count += 1
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        print(f"\n{'='*80}")
        print(f"🔄 WORKFLOW STEP {self.step_count} | {timestamp}")
        print(f"🎯 STAGE: {stage_name.upper()}")
        print(f"🤖 AGENT: {agent_name}")
        if description:
            print(f"📝 PURPOSE: {description}")
        print(f"{'='*80}")
        
        self.stage_history.append({
            'step': self.step_count,
            'stage': stage_name,
            'agent': agent_name,
            'timestamp': timestamp
        })
    
        
    def print_workflow_summary(self):
        print(f"\n{'🎯 WORKFLOW EXECUTION SUMMARY':=^80}")
        for entry in self.stage_history:
            print(f"Step {entry['step']:2d} | {entry['timestamp']} | {entry['stage']:12s} | {entry['agent']}")
        print(f"{'='*80}")


class EnhancedRoleSystem:
    def __init__(self):
        self.visualizer = WorkflowVisualizer()
        self.complexity_level = "simple"  # simple or complex
        self.analysis_topic = ""
        self.current_date = datetime.now().strftime('%B %d, %Y')
        
    def set_complexity(self, level: str):
        """Set workflow complexity: 'simple' or 'complex'"""
        self.complexity_level = level.lower()
        
    def set_analysis_topic(self, topic: str):
        """Set the topic/subject matter for analysis"""
        self.analysis_topic = topic
        
    def _is_time_sensitive_topic(self, topic: str) -> bool:
        """Determine if a topic requires current/recent information"""
        time_sensitive_keywords = [
            "conflict", "war", "news", "latest", "current", "recent", "update", 
            "breaking", "trend", "market", "economy", "politics", "election",
            "crisis", "emergency", "outbreak", "development", "breakthrough"
        ]
        topic_lower = topic.lower()
        return any(keyword in topic_lower for keyword in time_sensitive_keywords)
        
    def _generate_role_definition(self, role: str) -> str:
        """Generate role definition dynamically based on topic"""
        is_time_sensitive = self._is_time_sensitive_topic(self.analysis_topic)
        time_context = f"CRITICAL: Today is {self.current_date}. Focus on current/recent information." if is_time_sensitive else ""
        
        base_templates = {
            "collector": f"""You are an INFORMATION COLLECTOR specialized in gathering information about {self.analysis_topic}.

{time_context}

YOUR TASK: Use your web search tools to find comprehensive information. For each search:
1. Use search_web_information() with relevant search terms
2. If you find specific URLs in results, use scrape_webpage_content() to get detailed information
3. Search from multiple angles to get complete coverage

Focus on: Key facts and figures, credible sources, official statements, and relevant developments.
Always cite your sources and organize findings clearly.""",

            "analyzer": f"""You are a PATTERN ANALYZER for {self.analysis_topic}.

{time_context}

Focus on: 
1. Identifying patterns and trends in the collected information
2. Evaluating source credibility and reliability
3. Finding cause-and-effect relationships
4. Assessing broader implications and context
5. Providing structured insights with supporting evidence

Use verify_information_with_search() to cross-check critical claims that need verification.
Your analysis should be thorough and evidence-based.""",

            "synthesizer": f"""You are a KNOWLEDGE SYNTHESIZER creating a comprehensive analysis of {self.analysis_topic}.

{time_context}

Create a well-structured comprehensive report that integrates all findings. Your report should be:
- Complete and thorough
- Well-organized with clear sections
- Evidence-based with proper citations
- Actionable for decision-makers
- Appropriate for the complexity and nature of the topic

Structure your report in a way that best serves the topic and audience needs.""",

            "validator": f"""You are a QUALITY VALIDATOR for {self.analysis_topic} analysis.

{time_context}

Focus on: 
1. Cross-checking facts against reliable sources
2. Ensuring logical consistency and accuracy
3. Identifying potential biases or gaps
4. Evaluating completeness of coverage
5. Suggesting improvements for accuracy and clarity

Use verify_information_with_search() to fact-check critical claims. Ensure the analysis meets high quality standards.""",

            "communicator": f"""You are a PROFESSIONAL ANALYST creating the definitive report on {self.analysis_topic}.

{time_context}

Create a polished, professional report that effectively communicates the analysis findings. 
Your report should:
- Be clear and well-structured
- Include an executive summary
- Present findings logically
- Provide actionable insights
- Be appropriate for professional audiences
- Follow best practices for analytical reporting

Focus on clarity, accuracy, and practical value."""
        }
        
        return base_templates.get(role, f"You are a {role} analyzing {self.analysis_topic}.")
        
    def create_role_agents(self, client, selected_roles: List[str]) -> Dict[str, AssistantAgent]:
        agents = {}
        
        for role in selected_roles:
            # Generate role definition dynamically
            role_definition = self._generate_role_definition(role)
            
            # Define tools for each agent type
            tools = []
            if role == "collector":
                tools = [search_web_information, scrape_webpage_content]
            elif role == "analyzer":
                tools = [verify_information_with_search, search_web_information]
            elif role == "validator":
                tools = [verify_information_with_search, search_web_information]
            
            agents[role] = AssistantAgent(
                name=role,
                model_client=client,
                system_message=role_definition,
                tools=tools if tools else None
            )
                
        return agents

    def create_enhanced_workflow(self, agents: Dict[str, AssistantAgent], task: str) -> GraphFlow:
        filtered_agents = {}
        
        if self.complexity_level == "simple":
            # Simple workflow: collector -> analyzer -> synthesizer (final)
            agent_list = ["collector", "analyzer", "synthesizer"]
        else:
            # Complex workflow: collector -> analyzer -> synthesizer -> validator -> communicator (final)
            agent_list = ["collector", "analyzer", "synthesizer", "validator", "communicator"]
        
        # Filter to only include agents we need
        selected_agents = {role: agents[role] for role in agent_list if role in agents}
        
        for i, (role, agent) in enumerate(selected_agents.items()):
            if i == 0:
                filtered_agents[role] = agent
            else:
                # Different filtering strategy for communicator vs other agents
                if role == "communicator":
                    # Communicator should see original task + synthesizer output + validator feedback
                    filters = [
                        PerSourceFilter(source="user", position="first", count=1),  # Original task
                        PerSourceFilter(source="synthesizer", position="last", count=1),  # Main content
                        PerSourceFilter(source="validator", position="last", count=1)    # Quality feedback
                    ]
                else:
                    # Other agents see previous agent's output
                    prev_sources = agent_list[:i]
                    filters = [PerSourceFilter(source=src, position="last", count=1) for src in prev_sources]
                
                filtered_agents[role] = MessageFilterAgent(
                    name=role,
                    wrapped_agent=agent,
                    filter=MessageFilterConfig(per_source=filters)
                )
        
        builder = DiGraphBuilder()
        for role in agent_list:
            if role in filtered_agents:
                builder.add_node(filtered_agents[role])
            
        for i in range(len(agent_list) - 1):
            if agent_list[i] in filtered_agents and agent_list[i + 1] in filtered_agents:
                builder.add_edge(agent_list[i], agent_list[i + 1])
            
        return GraphFlow(
            participants=list(filtered_agents.values()),
            graph=builder.build(),
            termination_condition=MaxMessageTermination(len(agent_list) + 2)  # Add buffer to ensure completion
        )

    async def run_enhanced_workflow(self, workflow: GraphFlow, task: str):
        print(f"\n{'🚀 STARTING ENHANCED WORKFLOW EXECUTION':=^80}")
        print(f"📋 Task: {task}")
        print(f"🎯 Complexity: {self.complexity_level.upper()} workflow")
        if self.complexity_level == "simple":
            print("📝 Steps: collector → analyzer → synthesizer (final report)")
        else:
            print("📝 Steps: collector → analyzer → synthesizer → validator → communicator (final report)")
        print(f"{'='*80}")
        
        stage_outputs = {}
        final_result = ""
        final_agent = "synthesizer" if self.complexity_level == "simple" else "communicator"
        expected_steps = 3 if self.complexity_level == "simple" else 5
        
        workflow_completed = False
        agent_final_responses = set()  # Track which agents have given final responses
        
        try:
            async for event in workflow.run_stream(task=task):
                if hasattr(event, 'source') and hasattr(event, 'content'):
                    agent_name = event.source
                    content = event.content
                    
                    if agent_name in ["user", "DiGraphStopAgent"]:
                        continue
                    
                    # Check if this is a tool call request or execution event (skip logging these)
                    event_type = getattr(event, 'type', '')
                    if event_type in ['ToolCallRequestEvent', 'ToolCallExecutionEvent']:
                        print(f"🔧 {agent_name} using tools...")
                        continue
                        
                    # Only log final responses from agents (TextMessage or main chat responses)
                    if event_type in ['TextMessage', ''] and agent_name not in agent_final_responses:
                        stage_name = self._get_stage_name(agent_name)
                        framework = ROLE_FRAMEWORKS.get(agent_name)
                        description = f"Cognitive focus: {framework.cognitive_dimensions[0].value}" if framework else ""
                        self.visualizer.print_stage_header(stage_name, agent_name, description)
                        
                        agent_final_responses.add(agent_name)  # Mark this agent as having provided final response
                        
                        # Print a preview of the content for intermediate agents
                        if agent_name != final_agent:
                            preview = content[:200] + "..." if len(content) > 200 else content
                            print(f"📄 {agent_name.title()} Output Preview: {preview}")
                    
                    # Check if this is the final agent for this workflow complexity
                    if agent_name == final_agent and agent_name not in stage_outputs:
                        print(f"\n{'📋 FINAL ANALYSIS REPORT':=^80}")
                        print(content)
                        print(f"{'='*80}")
                        final_result = content
                        workflow_completed = True
                    
                    stage_outputs[agent_name] = content
                    
                    # Break immediately after final agent completes
                    if workflow_completed:
                        break
                    
                elif hasattr(event, 'messages') and not isinstance(event.messages, list):
                    # This is the TaskResult - workflow is complete
                    workflow_completed = True
                    break
                    
        except asyncio.CancelledError:
            print("🔄 Workflow execution cancelled - completing gracefully...")
        except Exception as e:
            print(f"⚠️  Workflow execution completed")
        
        # Ensure we have the final result
        if not workflow_completed and final_agent in stage_outputs:
            final_result = stage_outputs[final_agent]
            workflow_completed = True
        
        # Check if we completed the expected workflow
        if len(stage_outputs) < expected_steps and not workflow_completed:
            print(f"⚠️  Warning: Expected {expected_steps} steps but only completed {len(stage_outputs)}")
        
        self.visualizer.print_workflow_summary()
        
        return final_result
        
    def _get_stage_name(self, agent_name: str) -> str:
        stage_map = {
            "collector": "INFORMATION_GATHERING",
            "analyzer": "PATTERN_ANALYSIS", 
            "synthesizer": "KNOWLEDGE_SYNTHESIS",
            "validator": "QUALITY_VALIDATION",
            "communicator": "FINAL_REPORT_GENERATION"
        }
        return stage_map.get(agent_name, agent_name.upper())


async def check_tools_availability():
    """Check if the required tools are available and properly configured."""
    print("🔧 Checking tool availability...")
    
    # Check Anthropic API key
    anthropic_available = anthropic_search_module.check_api_key()
    print(f"🔍 Anthropic Search Tool: {'✅ Available' if anthropic_available else '❌ Missing API Key'}")
    
    # Check web scraper health
    scraper_health = await web_scraper_module.check_scraper_health()
    scraper_available = scraper_health["node_js_available"] and scraper_health["script_exists"]
    print(f"🌐 Web Scraper Tool: {'✅ Available' if scraper_available else '❌ Missing Dependencies'}")
    
    if not anthropic_available:
        print("⚠️  Warning: Set ANTHROPIC_API_KEY environment variable for search functionality")
    
    if not scraper_available:
        print("⚠️  Warning: Ensure Node.js is installed and scraper script exists")
    
    return anthropic_available or scraper_available


async def main():
    load_dotenv()
    TOGETHER_API_KEY = os.getenv('TOGETHER_API_KEY')

    print("🤖 Enhanced Agent Workflow System with Web Intelligence Tools")
    print("🎯 Automated Analysis Pipeline")
    print(f"📅 Current Date: {datetime.now().strftime('%B %d, %Y')}")
    
    # Check tool availability
    tools_available = await check_tools_availability()
    if not tools_available:
        print("❌ No web tools available. Continuing with limited functionality...")
    
    # Simple user input - just the topic
    print("\n" + "="*60)
    print("📝 ANALYSIS SETUP")
    print("="*60)
    
    topic = input("Enter your custom topic: ").strip()
    
    if not topic:
        topic = "Russia Ukraine conflict"
        
    # Get complexity level
    complexity = input("Enter complexity level (simple/complex) [simple]: ").strip().lower() or "simple"
    
    # Setup the role system
    role_system = EnhancedRoleSystem()
    role_system.set_complexity(complexity)
    role_system.set_analysis_topic(topic)
    
    # Create model client
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
    
    all_roles = ["collector", "analyzer", "synthesizer", "validator", "communicator"]
    
    print(f"\n🔧 Creating specialized analysis agents for: {topic}")
    agents = role_system.create_role_agents(client, all_roles)
    print(f"✅ Created {len(agents)} specialized agents with web intelligence tools")
    
    # Create workflow
    workflow = role_system.create_enhanced_workflow(agents, topic)
    
    # Generate analysis task
    is_time_sensitive = role_system._is_time_sensitive_topic(topic)
    time_instruction = f"CRITICAL: Today is {datetime.now().strftime('%B %d, %Y')}. Use web search tools to find the most recent information. Do NOT rely on outdated training data." if is_time_sensitive else "Use web search tools to gather comprehensive information on this topic."
    
    analysis_task = f"""ANALYSIS REQUEST

Provide a comprehensive analysis of: {topic}

{time_instruction}

Focus on:
1. Thorough information gathering from reliable sources
2. Analysis of key patterns, trends, and implications  
3. Assessment of significance and impact
4. Well-structured presentation of findings
5. Actionable insights and conclusions

Use your specialized tools to gather and verify information as needed."""
    
    try:
        print(f"\n🚀 Starting {complexity} analysis of: {topic}")
        final_result = await role_system.run_enhanced_workflow(workflow, analysis_task)
        
        print(f"\n{'📄 ANALYSIS COMPLETE':=^80}")
        print(f"💡 {complexity.title()} analysis pipeline executed successfully!")
        
    except KeyboardInterrupt:
        print(f"\n{'⏹️  ANALYSIS INTERRUPTED':=^80}")
        print("💡 Gracefully shutting down...")
        
    except Exception as e:
        print(f"\n{'⚠️  ANALYSIS ERROR':=^80}")
        print(f"💡 Analysis completed with issues: {str(e)}")
        
    finally:
        # Ensure clean shutdown
        print("🔄 Cleaning up resources...")
        await asyncio.sleep(0.1)  # Brief pause for cleanup

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Application error: {e}")
    finally:
        print("🏁 Application shutdown complete.")