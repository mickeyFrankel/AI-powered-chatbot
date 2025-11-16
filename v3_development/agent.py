"""Agent System with Thinking Display"""
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain.callbacks.base import BaseCallbackHandler
from typing import Any, Dict, List, Callable, Optional

from config import APIKeys, AgentConfig
from database import VectorDatabase
from tools import ChatbotTools


class StreamingThinkingCallback(BaseCallbackHandler):
    """Callback that streams thinking to UI in real-time"""
    
    def __init__(self, stream_func: Optional[Callable] = None):
        """
        Initialize callback.
        
        Args:
            stream_func: Function to call with each thought event
                        If None, prints to console
        """
        super().__init__()
        self.stream_func = stream_func or self._print_to_console
        self.thoughts = []
        self.step = 0
    
    def _print_to_console(self, event: Dict):
        """Default: print to console"""
        emoji = event.get('emoji', '')
        msg = event.get('message', '')
        print(f"  {emoji} {msg}")
    
    def on_agent_action(self, action: Any, **kwargs):
        """Called when agent decides on action"""
        self.step += 1
        event = {
            'type': 'thinking',
            'step': self.step,
            'tool': action.tool,
            'message': f"Using tool: {action.tool}",
            'emoji': '🤔'
        }
        self.thoughts.append(event)
        self.stream_func(event)
    
    def on_tool_end(self, output: str, **kwargs):
        """Called when tool completes"""
        event = {
            'type': 'tool_result',
            'step': self.step,
            'message': 'Got results',
            'emoji': '✅'
        }
        self.thoughts.append(event)
        self.stream_func(event)
    
    def on_agent_finish(self, finish: Any, **kwargs):
        """Called when agent finishes"""
        event = {
            'type': 'finish',
            'step': self.step,
            'message': 'Formulating final answer',
            'emoji': '💡'
        }
        self.thoughts.append(event)
        self.stream_func(event)
    
    def get_thoughts(self) -> List[Dict]:
        """Get all captured thoughts"""
        return self.thoughts
    
    def display_summary(self):
        """Display thinking summary (for console mode)"""
        if self.thoughts:
            print("\n" + "="*80)
            print("🧠 AGENT THINKING:")
            for thought in self.thoughts:
                print(f"  {thought['emoji']} {thought['message']}")
            print("="*80 + "\n")


class ChatAgent:
    """Main agent with thinking display - Facade Pattern"""
    
    def __init__(self, db: VectorDatabase):
        self.db = db
        self.llm = ChatOpenAI(
            model=AgentConfig.LLM_MODEL,
            api_key=APIKeys.get_openai_key(),
            temperature=AgentConfig.LLM_TEMPERATURE
        )
        self.chat_history = []
        self.tools_factory = ChatbotTools(db)
    
    def answer(self, user_input: str, stream_func: Optional[Callable] = None) -> str:
        """
        Answer with optional streaming thinking.
        
        Args:
            user_input: User's question
            stream_func: Optional function to stream thinking events
                        If None, prints to console
        
        Returns:
            Bot's answer
        """
        tools = self.tools_factory.create_tools()
        prompt = self._create_prompt()
        
        agent = create_openai_functions_agent(
            llm=self.llm,
            tools=tools,
            prompt=prompt
        )
        
        # Create streaming callback
        thinking = StreamingThinkingCallback(stream_func)
        
        executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=False,
            max_iterations=AgentConfig.MAX_ITERATIONS,
            handle_parsing_errors=True,
            callbacks=[thinking]
        )
        
        response = executor.invoke({
            "input": user_input,
            "chat_history": self.chat_history
        })
        
        # Display summary if console mode
        if stream_func is None:
            thinking.display_summary()
        
        # Update history
        self.chat_history.append(HumanMessage(content=user_input))
        self.chat_history.append(AIMessage(content=response['output']))
        self._trim_history()
        
        return response['output']
    
    def _create_prompt(self):
        return ChatPromptTemplate.from_messages([
            ("system", """You are an intelligent contact database assistant.

**TOOL SELECTION RULES:**

1. **Profession/Company/Keyword queries** → USE search_keyword
   - Pattern: "עובדים ב-X", "works at X", "כל רופאים", "all lawyers"
   - Keywords: עובדים, רופא, עורך דין, משגיח, ועד בית, company names
   - Example: "אנשים ב-AI" → search_keyword("AI", 100)

2. **Person names** → USE search_semantic
   - Pattern: "דוד כהן", "find Sarah", "מי זה יוסי"
   - Example: "דוד" → search_semantic("דוד", 5)

3. **Count queries** → USE count_documents or count_by_language
   - Pattern: "כמה אנשים", "how many contacts"

4. **Alphabetical** → USE list_by_prefix
   - Pattern: "names starting with A"

**CRITICAL:** When user says "עובדים ב-" or "works at", ALWAYS use search_keyword, NEVER search_semantic!

Be concise. Show phone numbers when available."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
    
    def _trim_history(self):
        max_messages = AgentConfig.MAX_HISTORY_TURNS * 2
        if len(self.chat_history) > max_messages:
            self.chat_history = self.chat_history[-max_messages:]
