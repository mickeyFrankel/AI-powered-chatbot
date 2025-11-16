"""
Thinking Callback Handler for Agent Reasoning Display

Captures and streams agent reasoning, tool calls, and observations
to show the user how the bot thinks.

Author: AI Assistant  
Date: 2025-10-30
"""

from typing import Any, Dict, List
from langchain.callbacks.base import BaseCallbackHandler
import json


class ThinkingCallbackHandler(BaseCallbackHandler):
    """
    Callback handler that captures agent thinking steps.
    
    This handler intercepts agent actions, tool calls, and observations
    to provide transparency into the agent's decision-making process.
    """
    
    def __init__(self):
        """Initialize with empty thought list."""
        self.thoughts: List[Dict[str, str]] = []
        self.current_step = 0
    
    def on_agent_action(self, action: Any, **kwargs) -> None:
        """Called when agent decides on an action."""
        self.current_step += 1
        
        # Extract tool and reasoning
        tool_name = action.tool
        tool_input = action.tool_input
        reasoning = action.log if hasattr(action, 'log') else ""
        
        # Parse reasoning to extract thought
        thought = self._extract_thought(reasoning)
        
        self.thoughts.append({
            "type": "thinking",
            "step": self.current_step,
            "thought": thought,
            "tool": tool_name,
            "input": str(tool_input),
            "emoji": "🤔"
        })
    
    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs
    ) -> None:
        """Called when tool execution starts."""
        tool_name = serialized.get("name", "unknown")
        
        self.thoughts.append({
            "type": "tool_start",
            "step": self.current_step,
            "message": f"Using tool: {tool_name}",
            "tool": tool_name,
            "emoji": "🔧"
        })
    
    def on_tool_end(self, output: str, **kwargs) -> None:
        """Called when tool execution completes."""
        # Truncate long outputs
        display_output = output[:200] + "..." if len(output) > 200 else output
        
        self.thoughts.append({
            "type": "tool_result",
            "step": self.current_step,
            "message": "Got results",
            "output": display_output,
            "emoji": "✅"
        })
    
    def on_tool_error(self, error: Exception, **kwargs) -> None:
        """Called when tool execution fails."""
        self.thoughts.append({
            "type": "error",
            "step": self.current_step,
            "message": f"Error: {str(error)}",
            "emoji": "❌"
        })
    
    def on_agent_finish(self, finish: Any, **kwargs) -> None:
        """Called when agent completes reasoning."""
        self.thoughts.append({
            "type": "finish",
            "step": self.current_step,
            "message": "Formulating final answer",
            "emoji": "💡"
        })
    
    def _extract_thought(self, log: str) -> str:
        """
        Extract the thought/reasoning from agent log.
        
        Args:
            log: Raw agent log string
            
        Returns:
            Cleaned thought text
        """
        if not log:
            return "Analyzing query..."
        
        # Common patterns in agent logs
        if "Thought:" in log:
            thought = log.split("Thought:")[1].split("Action:")[0].strip()
            return thought
        
        # Fallback: first sentence
        lines = log.strip().split('\n')
        if lines:
            return lines[0][:200]
        
        return "Processing..."
    
    def get_thoughts(self) -> List[Dict[str, str]]:
        """
        Get all captured thoughts.
        
        Returns:
            List of thought dictionaries
        """
        return self.thoughts
    
    def clear(self) -> None:
        """Clear all thoughts."""
        self.thoughts = []
        self.current_step = 0
    
    def to_markdown(self) -> str:
        """
        Format thoughts as markdown for display.
        
        Returns:
            Markdown formatted string
        """
        lines = ["## 🧠 Agent Thinking Process\n"]
        
        for thought in self.thoughts:
            emoji = thought.get("emoji", "")
            step = thought.get("step", "")
            
            if thought["type"] == "thinking":
                lines.append(f"**Step {step}:** {emoji} {thought['thought']}")
                lines.append(f"  → Using tool: `{thought['tool']}`")
            
            elif thought["type"] == "tool_start":
                lines.append(f"{emoji} {thought['message']}")
            
            elif thought["type"] == "tool_result":
                lines.append(f"{emoji} {thought['message']}")
            
            elif thought["type"] == "error":
                lines.append(f"{emoji} {thought['message']}")
            
            elif thought["type"] == "finish":
                lines.append(f"\n{emoji} {thought['message']}")
            
            lines.append("")  # Empty line between steps
        
        return "\n".join(lines)
