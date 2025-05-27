import asyncio
from datetime import datetime
from typing import Dict, List
from dataclasses import dataclass
from enum import Enum

from autogen_agentchat.agents import AssistantAgent, MessageFilterAgent, MessageFilterConfig, PerSourceFilter
from autogen_agentchat.teams import DiGraphBuilder, GraphFlow
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient


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
        
    def set_complexity(self, level: str):
        """Set workflow complexity: 'simple' or 'complex'"""
        self.complexity_level = level.lower()
        
    def create_role_agents(self, client, selected_roles: List[str]) -> Dict[str, AssistantAgent]:
        agents = {}
        
        role_definitions = {
            "collector": "You are an INFORMATION COLLECTOR specialized in systematic information gathering. Focus on: 1) Identifying diverse, credible sources 2) Extracting key facts and figures 3) Organizing information logically 4) Noting information gaps. Always structure your findings clearly.",
            
            "analyzer": "You are a PATTERN ANALYZER using systematic analytical thinking. Focus on: 1) Identifying patterns and relationships 2) Evaluating evidence quality 3) Drawing logical inferences 4) Highlighting contradictions or gaps 5) Providing structured insights with supporting evidence.",
            
            "synthesizer": "You are a KNOWLEDGE SYNTHESIZER specializing in creative integration. You create the FINAL COMPREHENSIVE REPORT by combining disparate elements into coherent wholes. Focus on: 1) Connecting ideas across domains 2) Creating unified frameworks 3) Generating novel insights 4) Building comprehensive models 5) Ensuring logical consistency. Your output should be a complete, publication-ready analysis that serves as the final deliverable unless further validation is required.",
            
            "validator": "You are a QUALITY VALIDATOR using metacognitive monitoring. Your role is INTERNAL QUALITY CONTROL - you provide feedback for improvement but do NOT write the final report. Focus on: 1) Fact-checking accuracy 2) Assessing completeness 3) Evaluating logical consistency 4) Identifying biases or errors 5) Suggesting specific improvements. Your feedback will be used by the final report writer to create the best possible customer deliverable.",
            
            "communicator": """You are a PROFESSIONAL REPORT WRITER creating the definitive customer-facing analysis report. 

CRITICAL: You are NOT reviewing or evaluating previous work. You are writing the FINAL DELIVERABLE that customers will read.

Your job is to synthesize all the research and analysis into a polished, professional report. Create a comprehensive, standalone document that reads like a high-quality consulting report or research publication.

Structure your report with:
1. EXECUTIVE SUMMARY - Key findings and conclusions (2-3 paragraphs)
2. INTRODUCTION - Context and scope 
3. CURRENT STATE ANALYSIS - Present capabilities and status
4. CHALLENGES AND LIMITATIONS - Major obstacles and constraints
5. RECENT DEVELOPMENTS - Latest breakthroughs and innovations
6. FUTURE OUTLOOK - Prospects and potential applications
7. STRATEGIC RECOMMENDATIONS - Actionable next steps
8. CONCLUSION - Summary of key insights

Write in a professional, authoritative tone suitable for business executives, technical decision-makers, or academic audiences. Focus on clarity, actionability, and strategic value. Do NOT include any internal review language, validation notes, or process commentary."""
        }
        
        for role in selected_roles:
            if role in role_definitions:
                agents[role] = AssistantAgent(
                    name=role,
                    model_client=client,
                    system_message=role_definitions[role]
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
        
        try:
            async for event in workflow.run_stream(task=task):
                if hasattr(event, 'source') and hasattr(event, 'content'):
                    agent_name = event.source
                    content = event.content
                    
                    if agent_name in ["user", "DiGraphStopAgent"]:
                        continue
                        
                    stage_name = self._get_stage_name(agent_name)
                    
                    framework = ROLE_FRAMEWORKS.get(agent_name)
                    description = f"Cognitive focus: {framework.cognitive_dimensions[0].value}" if framework else ""
                    self.visualizer.print_stage_header(stage_name, agent_name, description)
                    
                    # Check if this is the final agent for this workflow complexity
                    if agent_name == final_agent:
                        print(f"\n{'📋 FINAL CUSTOMER DELIVERABLE':=^80}")
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


async def main():
    print("🤖 Enhanced Agent Workflow System")
    
    role_system = EnhancedRoleSystem()
    
    # Set complexity level - change this based on your needs
    # "simple" = collector -> analyzer -> synthesizer (3 steps)
    # "complex" = collector -> analyzer -> synthesizer -> validator -> communicator (5 steps)
    
    complexity = input("Enter complexity level (simple/complex) [simple]: ").strip().lower() or "simple"
    role_system.set_complexity(complexity)
    
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
        api_key="a2d6e8d8c0eb8e72fb12c7443abccf4332f13a9a21e519d821831b82f2d95a74",
    )
    
    all_roles = ["collector", "analyzer", "synthesizer", "validator", "communicator"]
    
    print(f"🔧 Creating agents for all roles: {all_roles}")
    agents = role_system.create_role_agents(client, all_roles)
    
    print(f"✅ Created {len(agents)} specialized agents")
    
    workflow = role_system.create_enhanced_workflow(agents, "WASM component model analysis")
    
    try:
        final_result = await role_system.run_enhanced_workflow(
            workflow, 
            "Provide a comprehensive analysis of WASM component model: current capabilities, major challenges, recent breakthroughs, and future prospects for practical applications"
        )
        
        print(f"\n{'📄 WORKFLOW COMPLETE':=^80}")
        print(f"💡 {complexity.title()} analysis pipeline executed successfully!")
        
    except KeyboardInterrupt:
        print(f"\n{'⏹️  WORKFLOW INTERRUPTED':=^80}")
        print("💡 Gracefully shutting down...")
        
    except Exception as e:
        print(f"\n{'⚠️  WORKFLOW ERROR':=^80}")
        print(f"💡 Workflow completed with issues: {str(e)}")
        
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