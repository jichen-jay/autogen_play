"""
Cognitive Workflow System - Fixed version
Addresses runtime shutdown issues, race conditions, and asyncio.wait_for usage
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional

from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import GraphFlow, DiGraphBuilder
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.base import TaskResult

from cognitive_models import TaskFrame, TaskNature, ExecutionStrategy, ROLE_FRAMEWORKS
from cognitive_framer import MetaCognitiveFramer


class CognitiveWorkflowSystem:
    """
    Main cognitive workflow system with proper shutdown handling
    """
    
    def __init__(self):
        self.framer: Optional[MetaCognitiveFramer] = None
        self.execution_context = {}
        self.active_workflows = []
        
    def set_framer(self, framer: MetaCognitiveFramer):
        """Set the meta-cognitive framer"""
        self.framer = framer
        
    async def frame_and_plan_task(self, user_request: str) -> TaskFrame:
        """Frame the task using cognitive analysis"""
        if not self.framer:
            raise ValueError("Framer not set. Call set_framer() first.")
            
        task_frame = await self.framer.frame_task(user_request)
        
        # Display framing results
        print(f"\n{'🧠 COGNITIVE TASK FRAMING':=^80}")
        print(f"📋 Original Request: {task_frame.original_request}")
        print(f"🎯 User Intent: {task_frame.user_intent}")
        print(f"🔍 Task Nature: {task_frame.task_nature.value.upper()}")
        print(f"⚡ Execution Strategy: {task_frame.execution_strategy.value.upper()}")
        
        if task_frame.direct_action:
            print(f"🎯 Direct Action: {task_frame.direct_action}")
        
        print(f"🧠 Cognitive Demand: {task_frame.cognitive_demand.upper()}")
        print(f"🤖 Recommended Agents: {', '.join(task_frame.recommended_agents)}")
        print(f"🔄 Workflow Pattern: {task_frame.workflow_pattern}")
        print(f"✅ Success Criteria: {task_frame.success_criteria}")
        print(f"🤔 Reasoning: {task_frame.reasoning}")
        print("="*80)
        
        return task_frame
    
    def create_cognitive_agents(
        self, 
        client: OpenAIChatCompletionClient, 
        task_frame: TaskFrame,
        tool_functions: Dict
    ) -> Dict[str, AssistantAgent]:
        """Create specialized agents based on cognitive framing"""
        agents = {}
        
        for agent_name in task_frame.recommended_agents:
            if agent_name in ROLE_FRAMEWORKS:
                framework = ROLE_FRAMEWORKS[agent_name]
                
                # Create system message based on role framework and task context
                system_message = self._create_role_specific_system_message(
                    framework, task_frame, agent_name
                )
                
                # Create tools list for this agent
                agent_tools = []
                if agent_name == "collector":
                    agent_tools = [tool_functions.get("search_web_information")]
                elif agent_name == "analyzer":
                    agent_tools = [tool_functions.get("search_web_information")]
                
                # Filter out None tools
                agent_tools = [tool for tool in agent_tools if tool is not None]
                
                agents[agent_name] = AssistantAgent(
                    name=agent_name,
                    model_client=client,
                    tools=agent_tools,
                    system_message=system_message,
                    description=framework.name
                )
            else:
                # Fallback generic agent
                agents[agent_name] = AssistantAgent(
                    name=agent_name,
                    model_client=client,
                    system_message=f"You are a {agent_name} agent. Help with the task: {task_frame.original_request}",
                    description=f"Generic {agent_name} agent"
                )
        
        return agents
    
    def _create_role_specific_system_message(
        self, 
        framework, 
        task_frame: TaskFrame, 
        agent_name: str
    ) -> str:
        """Create role-specific system messages"""
        
        base_context = f"""
Current date: {datetime.now().strftime('%B %d, %Y')}
Task: {task_frame.original_request}
User Intent: {task_frame.user_intent}
Success Criteria: {task_frame.success_criteria}
"""
        
        if agent_name == "collector":
            return f"""You are an Information Collector agent. Your role is to gather relevant, accurate information.

{base_context}

Your specific responsibilities:
- Use available tools to search for current, relevant information
- Focus on factual, up-to-date data
- Prioritize authoritative sources
- Provide comprehensive information that addresses the user's needs
- Present information clearly and objectively

When using search tools, be strategic about your queries to get the most relevant results.
Always verify information from multiple sources when possible."""

        elif agent_name == "analyzer":
            return f"""You are a Pattern Analyzer agent. Your role is to analyze information and identify patterns, insights, and trends.

{base_context}

Your specific responsibilities:
- Analyze collected information for patterns and insights
- Identify key trends and relationships
- Provide context and interpretation
- Highlight important findings
- Use additional searches if needed to fill information gaps
- Present analysis in a structured, logical manner

Focus on extracting meaningful insights that help achieve the user's intent."""

        elif agent_name == "synthesizer":
            return f"""You are a Knowledge Synthesizer agent. Your role is to combine information from multiple sources into coherent insights.

{base_context}

Your specific responsibilities:
- Synthesize information from multiple sources
- Create comprehensive overviews
- Identify connections and relationships
- Resolve conflicting information
- Present integrated knowledge clearly
- Ensure completeness and accuracy

Focus on creating a unified understanding that addresses the user's needs."""

        elif agent_name == "validator":
            return f"""You are a Quality Validator agent. Your role is to verify accuracy and completeness.

{base_context}

Your specific responsibilities:
- Verify information accuracy
- Check for completeness against success criteria
- Identify potential gaps or issues
- Ensure quality standards are met
- Provide validation feedback
- Suggest improvements if needed

Focus on ensuring the final output meets the user's requirements."""

        elif agent_name == "communicator":
            return f"""You are a Final Report Generator agent. Your role is to create clear, comprehensive responses.

{base_context}

Your specific responsibilities:
- Create clear, well-structured final responses
- Ensure information is presented appropriately for the user
- Organize content logically
- Include all relevant information
- Use appropriate tone and style
- Meet the success criteria

Focus on creating a response that fully satisfies the user's request."""

        else:
            return f"""You are a {agent_name} agent helping with: {task_frame.original_request}

{base_context}

Provide helpful, accurate assistance to complete this task."""
    
    def create_adaptive_workflow(
        self, 
        agents: Dict[str, AssistantAgent], 
        task_frame: TaskFrame
    ) -> GraphFlow:
        """Create workflow based on execution strategy with proper termination"""
        
        builder = DiGraphBuilder()
        agent_list = list(agents.values())
        
        # Add all agents to the graph
        for agent in agent_list:
            builder.add_node(agent)
        
        # Build workflow topology based on strategy
        if task_frame.execution_strategy == ExecutionStrategy.DIRECT:
            # Simple single agent execution
            builder.set_entry_point(agent_list[0])
            
        elif task_frame.execution_strategy == ExecutionStrategy.DECOMPOSE:
            # Sequential multi-agent workflow
            for i in range(len(agent_list) - 1):
                builder.add_edge(agent_list[i], agent_list[i + 1])
            builder.set_entry_point(agent_list[0])
            
        elif task_frame.execution_strategy == ExecutionStrategy.ITERATIVE:
            # Sequential with potential loops (but we'll keep it simple)
            for i in range(len(agent_list) - 1):
                builder.add_edge(agent_list[i], agent_list[i + 1])
            builder.set_entry_point(agent_list[0])
            
        elif task_frame.execution_strategy == ExecutionStrategy.PARALLEL:
            # Fan-out from first agent
            if len(agent_list) > 1:
                for i in range(1, len(agent_list)):
                    builder.add_edge(agent_list[0], agent_list[i])
                builder.set_entry_point(agent_list[0])
            else:
                builder.set_entry_point(agent_list[0])
        
        # Create the workflow with proper termination
        # Use a reasonable but not too high max turn limit
        max_turns = min(10, len(agent_list) * 3)  # Reasonable limit
        termination = MaxMessageTermination(max_turns)
        
        workflow = GraphFlow(
            participants=agent_list,
            graph=builder.build(),
            termination_condition=termination,
            max_turns=max_turns
        )
        
        self.active_workflows.append(workflow)
        return workflow
    
    async def execute_cognitive_workflow(
        self, 
        workflow: GraphFlow, 
        task_frame: TaskFrame
    ) -> str:
        """Execute the cognitive workflow with proper error handling and cleanup"""
        
        print(f"\n{'🚀 EXECUTING COGNITIVE WORKFLOW':=^80}")
        print(f"📋 Task: {task_frame.original_request}")
        print(f"🎯 Intent: {task_frame.user_intent}")
        print(f"⚡ Strategy: {task_frame.execution_strategy.value.upper()}")
        print(f"📝 Agents: {', '.join(task_frame.recommended_agents)}")
        print("="*80)
        
        final_result = ""
        
        try:
            # Create the task message
            task_message = TextMessage(
                content=task_frame.original_request, 
                source="user"
            )
            
            # Execute workflow with timeout protection - FIXED
            messages = []
            task_completed = False
            
            try:
                # Create a timeout wrapper task
                async def run_workflow_with_timeout():
                    messages_list = []
                    async for message in workflow.run_stream(task=task_message):
                        if isinstance(message, TaskResult):
                            messages_list = message.messages
                            return messages_list, True
                        # Process intermediate messages if needed
                    return messages_list, False
                
                # Use asyncio.wait_for correctly with the wrapper
                messages, task_completed = await asyncio.wait_for(
                    run_workflow_with_timeout(), 
                    timeout=120.0  # 2 minute timeout
                )
                        
            except asyncio.TimeoutError:
                print("⏰ Workflow execution timed out, collecting available results...")
                task_completed = False
            
            # Extract the final result
            if messages:
                # Find the last meaningful message
                for msg in reversed(messages):
                    if hasattr(msg, 'content') and msg.content.strip():
                        if msg.source != 'user':  # Skip user messages
                            final_result = msg.content
                            break
                
                if not final_result:
                    final_result = "Task completed but no final result was generated."
            else:
                final_result = "No results were generated from the workflow."
            
            # Evaluation
            print(f"\n{'📊 EXECUTION EVALUATION':=^80}")
            print(f"✅ Success Criteria: {task_frame.success_criteria}")
            print(f"🎯 User Intent Fulfilled: {'Yes' if task_completed else 'Partial'}")
            
            print(f"\n{'🎯 WORKFLOW EXECUTION SUMMARY':=^80}")
            print("="*80)
            
        except Exception as e:
            print(f"⚠️ Workflow execution error: {e}")
            final_result = f"Workflow execution encountered an error: {str(e)}"
        
        finally:
            # Ensure proper cleanup
            await self._safe_cleanup_workflow(workflow)
        
        return final_result
    
    async def _safe_cleanup_workflow(self, workflow: GraphFlow):
        """Safely clean up workflow resources"""
        try:
            # Remove from active workflows
            if workflow in self.active_workflows:
                self.active_workflows.remove(workflow)
            
            # Try to gracefully stop the workflow
            if hasattr(workflow, '_runtime') and workflow._runtime:
                try:
                    # Check if runtime is still active before attempting shutdown
                    if hasattr(workflow._runtime, '_running') and workflow._runtime._running:
                        # Try to stop gracefully with timeout
                        await asyncio.wait_for(workflow._runtime.stop_when_idle(), timeout=5.0)
                except (asyncio.TimeoutError, Exception):
                    # Force stop if graceful stop fails
                    try:
                        if hasattr(workflow._runtime, 'stop'):
                            workflow._runtime.stop()
                    except:
                        pass  # Ignore force stop errors
            
            # Small delay to allow cleanup to complete
            await asyncio.sleep(0.1)
            
        except Exception as cleanup_error:
            # Don't raise cleanup errors, just log them
            print(f"⚠️ Cleanup warning (non-fatal): {cleanup_error}")
    
    async def cleanup_all_workflows(self):
        """Clean up all active workflows"""
        cleanup_tasks = []
        for workflow in self.active_workflows.copy():
            cleanup_tasks.append(self._safe_cleanup_workflow(workflow))
        
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        
        self.active_workflows.clear()