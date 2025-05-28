"""
Core data models and frameworks for the Cognitive Workflow System
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


class TaskNature(Enum):
    ATOMIC = "atomic"        # Single, direct task - execute literally
    COMPOSITE = "composite"  # Meta-task - needs decomposition into subtasks
    HYBRID = "hybrid"        # Contains both direct and meta elements


class ExecutionStrategy(Enum):
    DIRECT = "direct"           # Execute task as-is
    DECOMPOSE = "decompose"     # Break down into subtasks
    ITERATIVE = "iterative"     # Execute in phases with feedback
    PARALLEL = "parallel"       # Multiple independent subtasks


@dataclass
class SubTask:
    description: str
    priority: int  # 1 = highest
    dependencies: List[str]  # Names of subtasks this depends on
    estimated_effort: str    # low, medium, high
    requires_tools: List[str]  # Which tools needed


@dataclass
class TaskFrame:
    """Dynamic task framing based on LLM cognitive analysis"""
    original_request: str
    task_nature: TaskNature
    execution_strategy: ExecutionStrategy
    
    # For ATOMIC tasks
    direct_action: Optional[str] = None
    cognitive_demand: str = "medium"  # low, medium, high
    
    # For COMPOSITE/HYBRID tasks  
    subtasks: List[SubTask] = None
    integration_strategy: str = "sequential"  # sequential, parallel, hierarchical
    
    # Meta-cognitive insights
    user_intent: str = ""
    hidden_assumptions: List[str] = None
    success_criteria: str = ""
    
    # Adaptive guidance
    recommended_agents: List[str] = None
    workflow_pattern: str = "linear"  # linear, branching, cyclical
    
    reasoning: str = ""


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


# Pre-defined role frameworks
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