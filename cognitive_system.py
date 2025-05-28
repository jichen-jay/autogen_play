"""
Simplified Cognitive System focusing on task framing with AutoGen's built-in workflow patterns
"""

import asyncio
import os
from datetime import datetime
from typing import Dict, List
from dotenv import load_dotenv

from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat, SelectorGroupChat, Swarm
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination

import anthropic_search_module
import web_scraper_module
from cognitive_models import TaskFrame, TaskNature, ExecutionStrategy
from cognitive_framer import MetaCognitiveFramer


class CognitiveWorkflowSystem:
    """
    Simplified system that focuses on cognitive task framing 
    and uses AutoGen's built-in workflow patterns
    """
    
    def __init__(self):
        self.current_date = datetime.now().strftime('%B %d, %Y')
        
    def create_cognitive_agents(self, client: OpenAIChatCompletionClient, task_frame: TaskFrame, tool_functions: Dict) -> List[AssistantAgent]:
        """Create specialized agents based on task framing"""
        
        agents = []
        base_context = f"Today is {self.current_date}."
        
        # Create agents based on the task frame's recommended agents
        for role in task_frame.recommended_agents:
            
            if role == "collector":
                agent = AssistantAgent(
                    name="InformationCollector",
                    model_client=client,
                    system_message=f"""{base_context}

You are an INFORMATION COLLECTOR specialized in gathering relevant data.

TASK CONTEXT:
- Original Request: {task_frame.original_request}
- User Intent: {task_frame.user_intent}
- Success Criteria: {task_frame.success_criteria}

YOUR ROLE:
- Use search_web_information() for general web searches
- Use scrape_webpage_content() for specific URLs
- Gather comprehensive, relevant information
- Organize findings clearly with proper citations
- Focus on serving the user's true intent

When you have completed your information gathering, end your response with "INFORMATION_COLLECTED" to signal completion.""",
                    tools=[
                        tool_functions.get("search_web_information"),
                        tool_functions.get("scrape_webpage_content")
                    ]
                )
                agents.append(agent)
                
            elif role == "analyzer":
                agent = AssistantAgent(
                    name="PatternAnalyzer",
                    model_client=client,
                    system_message=f"""{base_context}

You are a PATTERN ANALYZER specialized in identifying insights and relationships.

TASK CONTEXT:
- Original Request: {task_frame.original_request}
- User Intent: {task_frame.user_intent}
- Success Criteria: {task_frame.success_criteria}

YOUR ROLE:
- Analyze information provided by the InformationCollector
- Identify key patterns, insights, and relationships
- Evaluate source credibility and evidence quality
- Use verify_information_with_search() when verification is needed
- Structure your analysis to advance the user's intent

When you have completed your analysis, end your response with "ANALYSIS_COMPLETE" to signal completion.""",
                    tools=[tool_functions.get("verify_information_with_search")]
                )
                agents.append(agent)
                
            elif role == "synthesizer":
                agent = AssistantAgent(
                    name="KnowledgeSynthesizer", 
                    model_client=client,
                    system_message=f"""{base_context}

You are a KNOWLEDGE SYNTHESIZER specialized in creating comprehensive responses.

TASK CONTEXT:
- Original Request: {task_frame.original_request}
- User Intent: {task_frame.user_intent}
- Success Criteria: {task_frame.success_criteria}

YOUR ROLE:
- Integrate information from InformationCollector and PatternAnalyzer
- Create well-structured, comprehensive content
- Ensure the response fulfills the user's intent and success criteria
- Present information clearly and logically
- Provide a complete answer to the original request

When you have completed your synthesis, end your response with "SYNTHESIS_COMPLETE" to signal completion."""
                )
                agents.append(agent)
                
            elif role == "validator":
                agent = AssistantAgent(
                    name="QualityValidator",
                    model_client=client,
                    system_message=f"""{base_context}

You are a QUALITY VALIDATOR specialized in ensuring accuracy and completeness.

TASK CONTEXT:
- Original Request: {task_frame.original_request}
- User Intent: {task_frame.user_intent}
- Success Criteria: {task_frame.success_criteria}

YOUR ROLE:
- Review the synthesized response for accuracy and completeness
- Cross-check critical facts and claims
- Verify the response meets the success criteria
- Use verify_information_with_search() for fact-checking if needed
- Identify any gaps or inconsistencies

If the response needs significant revision, say "NEEDS_REVISION" and explain why.
If the response is acceptable, say "VALIDATED" and provide your final assessment.""",
                    tools=[tool_functions.get("verify_information_with_search")]
                )
                agents.append(agent)
                
            elif role == "communicator":
                agent = AssistantAgent(
                    name="FinalCommunicator",
                    model_client=client,
                    system_message=f"""{base_context}

You are a PROFESSIONAL COMMUNICATOR specialized in creating polished final responses.

TASK CONTEXT:
- Original Request: {task_frame.original_request}
- User Intent: {task_frame.user_intent}
- Success Criteria: {task_frame.success_criteria}

YOUR ROLE:
- Create the final, polished response based on all previous work
- Present findings in the most effective format for the user
- Ensure clarity, accuracy, and professional quality
- Tailor communication style to the user's needs
- Confirm the response fulfills the original intent

Provide your final response and end with "TASK_COMPLETE" to signal completion."""
                )
                agents.append(agent)
        
        return agents
    
    def create_workflow_team(self, agents: List[AssistantAgent], task_frame: TaskFrame):
        """Create appropriate AutoGen team based on execution strategy"""
        
        # Create termination condition based on task nature
        if task_frame.task_nature == TaskNature.ATOMIC:
            # Simple tasks need fewer turns
            termination = MaxMessageTermination(len(agents) + 2)
        else:
            # Complex tasks may need more iterations
            completion_terms = TextMentionTermination("TASK_COMPLETE") | TextMentionTermination("VALIDATED")
            max_turns = MaxMessageTermination(20)  # Safety limit
            termination = completion_terms | max_turns
        
        # Choose team pattern based on execution strategy
        if task_frame.execution_strategy == ExecutionStrategy.DIRECT:
            # Single agent or simple round-robin
            return RoundRobinGroupChat(
                participants=agents,
                termination_condition=termination,
                max_turns=5
            )
            
        elif task_frame.execution_strategy == ExecutionStrategy.DECOMPOSE:
            # Sequential execution with clear handoffs
            return RoundRobinGroupChat(
                participants=agents,
                termination_condition=termination
            )
            
        elif task_frame.execution_strategy == ExecutionStrategy.ITERATIVE:
            # Use SelectorGroupChat for more dynamic control
            return SelectorGroupChat(
                participants=agents,
                model_client=agents[0]._model_client,  # Use same client
                termination_condition=termination,
                selector_prompt=f"""
You are managing a cognitive workflow for: "{task_frame.original_request}"

Current agents and their roles:
{self._format_agent_roles(agents)}

Based on the conversation history, select the next agent to speak.
Consider the user's intent: {task_frame.user_intent}

Return only the agent name."""
            )
            
        elif task_frame.execution_strategy == ExecutionStrategy.PARALLEL:
            # Use Swarm pattern for handoff-based coordination
            if len(agents) > 1:
                # Add handoff capabilities to agents
                for i, agent in enumerate(agents[:-1]):
                    next_agent = agents[i + 1]
                    agent._handoffs = [next_agent.name]
            
            return Swarm(
                participants=agents,
                termination_condition=termination
            )
        
        # Default fallback
        return RoundRobinGroupChat(
            participants=agents,
            termination_condition=termination
        )
    
    def _format_agent_roles(self, agents: List[AssistantAgent]) -> str:
        """Format agent descriptions for selector prompt"""
        role_descriptions = {
            "InformationCollector": "Gathers information using web search and scraping tools",
            "PatternAnalyzer": "Analyzes patterns and relationships in collected information", 
            "KnowledgeSynthesizer": "Integrates information into comprehensive responses",
            "QualityValidator": "Validates accuracy and completeness of responses",
            "FinalCommunicator": "Creates polished final responses"
        }
        
        formatted = []
        for agent in agents:
            desc = role_descriptions.get(agent.name, "Specialized cognitive agent")
            formatted.append(f"- {agent.name}: {desc}")
        
        return "\n".join(formatted)
    
    async def execute_cognitive_workflow(self, team, task_frame: TaskFrame):
        """Execute the workflow using AutoGen's built-in team execution"""
        
        print(f"\n{'🚀 EXECUTING COGNITIVE WORKFLOW':=^80}")
        print(f"📋 Task: {task_frame.original_request}")
        print(f"🎯 Intent: {task_frame.user_intent}")
        print(f"⚡ Strategy: {task_frame.execution_strategy.value.upper()}")
        print(f"🤖 Team Type: {type(team).__name__}")
        print(f"📝 Agents: {[agent.name for agent in team._participants]}")
        print(f"{'='*80}")
        
        try:
            # Execute the team workflow
            result = await team.run(task=task_frame.original_request)
            
            print(f"\n{'📄 WORKFLOW RESULTS':=^80}")
            
            # Display the conversation flow
            for i, message in enumerate(result.messages):
                if hasattr(message, 'source') and hasattr(message, 'content'):
                    if message.source != "user":
                        print(f"\n--- {message.source} ---")
                        # Show preview for non-final messages
                        if i < len(result.messages) - 1:
                            preview = message.content[:200] + "..." if len(message.content) > 200 else message.content
                            print(preview)
                        else:
                            # Show full final message
                            print(message.content)
            
            print(f"\n{'✅ COGNITIVE ANALYSIS COMPLETE':=^80}")
            print(f"💡 Task framed as: {task_frame.task_nature.value} → {task_frame.execution_strategy.value}")
            print(f"🎯 Success criteria: {task_frame.success_criteria}")
            print(f"{'='*80}")
            
            return result
            
        except Exception as e:
            print(f"⚠️  Workflow execution error: {e}")
            raise


# Tool Functions (same as before)
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


async def main():
    load_dotenv()
    TOGETHER_API_KEY = os.getenv('TOGETHER_API_KEY')

    print("🤖 Cognitive Task Framing with AutoGen Workflow Patterns")
    print("🧠 Meta-Cognitive Analysis + Built-in Team Execution")
    print(f"📅 Current Date: {datetime.now().strftime('%B %d, %Y')}")
    
    # User input
    print("\n" + "="*60)
    print("📝 COGNITIVE TASK INPUT")
    print("="*60)
    
    user_request = input("Enter your request: ").strip()
    if not user_request:
        user_request = "find what's on the first page of https://www.example.com/"
        print(f"Using default request: {user_request}")
    
    try:
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
        
        # Setup cognitive framing
        print(f"\n🧠 Analyzing task structure and intent...")
        framer = MetaCognitiveFramer(client)
        task_frame = await framer.frame_task(user_request)
        
        print(f"\n{'🧠 COGNITIVE TASK FRAMING':=^80}")
        print(f"📋 Original Request: {task_frame.original_request}")
        print(f"🎯 User Intent: {task_frame.user_intent}")
        print(f"🔍 Task Nature: {task_frame.task_nature.value.upper()}")
        print(f"⚡ Execution Strategy: {task_frame.execution_strategy.value.upper()}")
        print(f"🤖 Recommended Agents: {' → '.join(task_frame.recommended_agents)}")
        print(f"✅ Success Criteria: {task_frame.success_criteria}")
        print(f"🤔 Reasoning: {task_frame.reasoning}")
        print(f"{'='*80}")
        
        # Create tool functions
        tool_functions = {
            "search_web_information": search_web_information,
            "scrape_webpage_content": scrape_webpage_content,
            "verify_information_with_search": verify_information_with_search
        }
        
        # Create cognitive agents
        workflow_system = CognitiveWorkflowSystem()
        agents = workflow_system.create_cognitive_agents(client, task_frame, tool_functions)
        print(f"\n✅ Created {len(agents)} cognitive agents: {[agent.name for agent in agents]}")
        
        # Create appropriate AutoGen team
        team = workflow_system.create_workflow_team(agents, task_frame)
        print(f"✅ Created {type(team).__name__} team pattern")
        
        # Execute workflow
        result = await workflow_system.execute_cognitive_workflow(team, task_frame)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())