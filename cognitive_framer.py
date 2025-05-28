"""
Meta-cognitive task framing system that understands task structure and intent
Fixed version - addresses JSON parsing and model response handling for dict responses
"""

import json
from typing import Dict, Any
from autogen_ext.models.openai import OpenAIChatCompletionClient

from cognitive_models import (
    TaskFrame, TaskNature, ExecutionStrategy, SubTask
)


class MetaCognitiveFramer:
    """
    Advanced task framing system that understands task structure and intent
    """
    
    def __init__(self, client: OpenAIChatCompletionClient):
        self.client = client
        self.frame_cache = {}
        
    async def frame_task(self, user_request: str) -> TaskFrame:
        """
        Use LLM cognitive reasoning to understand and frame the task
        """
        
        # Check cache first
        request_key = user_request.lower().strip()
        if request_key in self.frame_cache:
            return self.frame_cache[request_key]
        
        framing_prompt = f"""You are a meta-cognitive task framer. Analyze this user request to understand its true structure and intent.

User Request: "{user_request}"

Provide your analysis in JSON format:

{{
    "task_nature": "atomic|composite|hybrid",
    "execution_strategy": "direct|decompose|iterative|parallel",
    "user_intent": "What the user really wants to achieve",
    "hidden_assumptions": ["assumption1", "assumption2"],
    "success_criteria": "How to know if the task is successfully completed",
    
    "direct_action": "For atomic tasks: the specific action to take",
    "cognitive_demand": "low|medium|high",
    
    "subtasks": [
        {{
            "description": "Specific subtask description",
            "priority": 1,
            "dependencies": ["other_subtask_names"],
            "estimated_effort": "low|medium|high",
            "requires_tools": ["tool_names"]
        }}
    ],
    "integration_strategy": "sequential|parallel|hierarchical",
    
    "recommended_agents": ["collector", "analyzer", "synthesizer", "validator", "communicator"],
    "workflow_pattern": "linear|branching|cyclical",
    
    "reasoning": "Detailed explanation of your framing analysis"
}}

GUIDELINES:

**ATOMIC Tasks** - Single, direct actions:
- "What is the capital of France?" 
- "Find the latest news about X"
- "Define artificial intelligence"

**COMPOSITE Tasks** - Meta-tasks requiring decomposition:
- "Analyze the impact of climate change on global economics"
- "Create a comprehensive business plan for a startup"
- "Compare and contrast multiple complex topics"

**HYBRID Tasks** - Mix of direct and meta elements:
- "What happened in Ukraine conflict and what are the implications?"
- "Explain quantum computing and its potential applications"

For COMPOSITE/HYBRID tasks, break them down into specific, actionable subtasks.
For ATOMIC tasks, identify the direct action needed.

Focus on understanding the USER'S TRUE INTENT, not just the literal words."""

        try:
            # Use the model client with json_output=True for structured response
            response = await self.client.create(
                messages=[{"role": "user", "content": framing_prompt}],
                json_output=True  # This should return structured data
            )
            
            # Handle the response content properly - FIXED FOR DICT RESPONSES
            frame_data = None
            response_content = response.content
            
            if isinstance(response_content, dict):
                # Direct dict response - this is what was causing the error
                frame_data = response_content
                print("✅ Received structured dict response from model")
                
            elif isinstance(response_content, str):
                # String response that needs JSON parsing
                try:
                    # Look for JSON block in the response
                    content_lines = response_content.strip()
                    
                    # Find JSON content between braces
                    start_idx = content_lines.find('{')
                    end_idx = content_lines.rfind('}')
                    
                    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                        json_content = content_lines[start_idx:end_idx + 1]
                        frame_data = json.loads(json_content)
                        print("✅ Parsed JSON from string response")
                    else:
                        print(f"⚠️  No JSON found in string response, using fallback")
                        return self._fallback_framing(user_request)
                        
                except json.JSONDecodeError as e:
                    print(f"⚠️  Failed to parse JSON response: {e}")
                    print(f"Raw response: {response_content[:500]}...")
                    return self._fallback_framing(user_request)
            else:
                print(f"⚠️  Unexpected response content type: {type(response_content)}")
                return self._fallback_framing(user_request)
            
            # Validate frame_data exists
            if not frame_data:
                print("⚠️  No frame data extracted, using fallback")
                return self._fallback_framing(user_request)
                
            # Validate required fields
            required_fields = ["task_nature", "execution_strategy", "user_intent", "success_criteria", "recommended_agents", "reasoning"]
            missing_fields = [field for field in required_fields if field not in frame_data]
            
            if missing_fields:
                print(f"⚠️  Missing required fields {missing_fields} in framing response, using fallback")
                return self._fallback_framing(user_request)
            
            # Create TaskFrame object with safe value extraction
            try:
                # Validate and clean enum values
                task_nature_val = str(frame_data["task_nature"]).lower().strip()
                if task_nature_val not in ["atomic", "composite", "hybrid"]:
                    print(f"⚠️  Invalid task_nature '{task_nature_val}', defaulting to 'hybrid'")
                    task_nature_val = "hybrid"
                
                execution_strategy_val = str(frame_data["execution_strategy"]).lower().strip()
                if execution_strategy_val not in ["direct", "decompose", "iterative", "parallel"]:
                    print(f"⚠️  Invalid execution_strategy '{execution_strategy_val}', defaulting to 'direct'")
                    execution_strategy_val = "direct"
                
                # Handle subtasks safely
                subtasks = None
                if frame_data.get("subtasks") and isinstance(frame_data["subtasks"], list):
                    try:
                        subtasks = []
                        for st in frame_data["subtasks"]:
                            if isinstance(st, dict):
                                # Ensure all required fields with defaults
                                subtask = SubTask(
                                    description=st.get("description", "No description"),
                                    priority=int(st.get("priority", 1)),
                                    dependencies=st.get("dependencies", []),
                                    estimated_effort=st.get("estimated_effort", "medium"),
                                    requires_tools=st.get("requires_tools", [])
                                )
                                subtasks.append(subtask)
                    except Exception as e:
                        print(f"⚠️  Error creating subtasks: {e}, skipping subtasks")
                        subtasks = None
                
                # Ensure recommended_agents is a list
                recommended_agents = frame_data.get("recommended_agents", ["collector"])
                if not isinstance(recommended_agents, list):
                    recommended_agents = ["collector"]
                
                # Ensure hidden_assumptions is a list
                hidden_assumptions = frame_data.get("hidden_assumptions", [])
                if not isinstance(hidden_assumptions, list):
                    hidden_assumptions = []
                
                task_frame = TaskFrame(
                    original_request=user_request,
                    task_nature=TaskNature(task_nature_val),
                    execution_strategy=ExecutionStrategy(execution_strategy_val),
                    direct_action=frame_data.get("direct_action"),
                    cognitive_demand=str(frame_data.get("cognitive_demand", "medium")).lower(),
                    subtasks=subtasks,
                    integration_strategy=str(frame_data.get("integration_strategy", "sequential")).lower(),
                    user_intent=str(frame_data["user_intent"]),
                    hidden_assumptions=hidden_assumptions,
                    success_criteria=str(frame_data["success_criteria"]),
                    recommended_agents=recommended_agents,
                    workflow_pattern=str(frame_data.get("workflow_pattern", "linear")).lower(),
                    reasoning=str(frame_data["reasoning"])
                )
                
                # Cache the frame
                self.frame_cache[request_key] = task_frame
                print("✅ Task frame created successfully")
                return task_frame
                
            except (ValueError, TypeError, KeyError) as e:
                print(f"⚠️  Failed to create TaskFrame from response: {e}")
                print(f"Frame data keys: {list(frame_data.keys()) if frame_data else 'None'}")
                return self._fallback_framing(user_request)
            
        except Exception as e:
            print(f"⚠️  Task framing failed, using fallback: {e}")
            import traceback
            traceback.print_exc()
            return self._fallback_framing(user_request)
    
    def _fallback_framing(self, user_request: str) -> TaskFrame:
        """Fallback framing using simple heuristics"""
        request_lower = user_request.lower()
        
        # Simple heuristics for task nature
        if any(word in request_lower for word in ["what is", "define", "who is", "when", "where"]):
            task_nature = TaskNature.ATOMIC
            strategy = ExecutionStrategy.DIRECT
            agents = ["collector"]
        elif any(word in request_lower for word in ["analyze", "compare", "evaluate", "assess"]):
            task_nature = TaskNature.COMPOSITE
            strategy = ExecutionStrategy.DECOMPOSE
            agents = ["collector", "analyzer", "synthesizer"]
        else:
            task_nature = TaskNature.HYBRID
            strategy = ExecutionStrategy.ITERATIVE
            agents = ["collector", "analyzer"]
        
        print(f"✅ Using fallback framing for task nature: {task_nature.value}")
        
        return TaskFrame(
            original_request=user_request,
            task_nature=task_nature,
            execution_strategy=strategy,
            direct_action=user_request if task_nature == TaskNature.ATOMIC else None,
            user_intent="User wants information or analysis",
            success_criteria="Provide relevant, accurate response",
            recommended_agents=agents,
            reasoning="Fallback heuristic-based framing"
        )
    
    async def adapt_frame_during_execution(self, current_frame: TaskFrame, execution_context: Dict) -> TaskFrame:
        """
        Dynamically adapt the task frame based on execution progress
        """
        
        adaptation_prompt = f"""You are monitoring task execution and may need to adapt the original plan.

Original Task: "{current_frame.original_request}"
Original Intent: "{current_frame.user_intent}"
Current Progress: {execution_context.get('progress_summary', 'No progress data')}

Should the task framing be adapted? Consider:
1. Is the original intent being fulfilled?
2. Have new subtasks or considerations emerged?
3. Is the execution strategy still optimal?
4. Should agents be added/removed from the workflow?

Respond in JSON:
{{
    "should_adapt": true/false,
    "adaptation_reason": "Why adaptation is needed",
    "new_subtasks": [{{...}}],
    "agent_changes": {{"add": ["agent_names"], "remove": ["agent_names"]}},
    "strategy_change": "new_strategy_if_changed"
}}"""

        try:
            response = await self.client.create(
                messages=[{"role": "user", "content": adaptation_prompt}],
                json_output=True
            )
            
            # Parse response similar to frame_task - handle both dict and string
            response_content = response.content
            adaptation = None
            
            if isinstance(response_content, dict):
                adaptation = response_content
            elif isinstance(response_content, str):
                start_idx = response_content.find('{')
                end_idx = response_content.rfind('}')
                
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    json_content = response_content[start_idx:end_idx + 1]
                    adaptation = json.loads(json_content)
                else:
                    return current_frame
            else:
                return current_frame
            
            if adaptation and adaptation.get("should_adapt", False):
                print(f"🧠 Adapting task frame: {adaptation.get('adaptation_reason', 'No reason provided')}")
                # Apply adaptations to the frame
                # This could modify subtasks, agents, strategy, etc.
                
            return current_frame  # Return adapted frame
            
        except Exception as e:
            print(f"⚠️  Frame adaptation failed: {e}")
            return current_frame