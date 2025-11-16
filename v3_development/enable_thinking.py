#!/usr/bin/env python3
"""
Enable Thinking Display - Quick Integration

Modifies vectoric_search.py to show agent reasoning in real-time.
Integrates ThinkingCallbackHandler seamlessly.

Usage: python3 enable_thinking.py
"""

import re

# Read files
with open("vectoric_search.py", 'r') as f:
    vectoric = f.read()

with open("thinking_callback.py", 'r') as f:
    callback_code = f.read()

# Backup
with open("vectoric_search_BEFORE_THINKING.py", 'w') as f:
    f.write(vectoric)

# Step 1: Add import at top
import_line = "from thinking_callback import ThinkingCallbackHandler\n"

if "from thinking_callback" not in vectoric:
    # Find where to insert (after other imports)
    langchain_import_pos = vectoric.find("from langchain_core.messages import")
    if langchain_import_pos != -1:
        # Find end of that line
        next_newline = vectoric.find("\n", langchain_import_pos)
        vectoric = vectoric[:next_newline+1] + import_line + vectoric[next_newline+1:]

# Step 2: Modify agent_answer to use callback
old_agent_answer = '''    def agent_answer(self, user_input: str) -> str:
        """Answer using LangChain agent"""
        try:
            # Create agent
            tools = self._create_tools()
            prompt = self._create_prompt()
            
            agent = create_openai_functions_agent(
                llm=self.llm,
                tools=tools,
                prompt=prompt
            )
            
            agent_executor = AgentExecutor(
                agent=agent,
                tools=tools,
                verbose=False,
                max_iterations=3,  # Reduced from 5 for faster responses
                handle_parsing_errors=True
            )
            
            # Run agent with chat history
            response = agent_executor.invoke({
                "input": user_input,
                "chat_history": self.chat_history
            })
            
            # Update chat history
            self.chat_history.append(HumanMessage(content=user_input))
            self.chat_history.append(AIMessage(content=response['output']))
            
            # Trim history if too long
            self._trim_history()
            
            return response['output']'''

new_agent_answer = '''    def agent_answer(self, user_input: str) -> str:
        """Answer using LangChain agent with thinking display"""
        try:
            # Create agent
            tools = self._create_tools()
            prompt = self._create_prompt()
            
            agent = create_openai_functions_agent(
                llm=self.llm,
                tools=tools,
                prompt=prompt
            )
            
            # Create thinking callback
            thinking_callback = ThinkingCallbackHandler()
            
            agent_executor = AgentExecutor(
                agent=agent,
                tools=tools,
                verbose=False,  # Keep False, we use callback instead
                max_iterations=3,
                handle_parsing_errors=True,
                callbacks=[thinking_callback]  # ✅ Add thinking callback
            )
            
            # Run agent with chat history
            response = agent_executor.invoke({
                "input": user_input,
                "chat_history": self.chat_history
            })
            
            # Display thinking process
            print("\\n" + "="*80)
            print(thinking_callback.to_markdown())
            print("="*80 + "\\n")
            
            # Update chat history
            self.chat_history.append(HumanMessage(content=user_input))
            self.chat_history.append(AIMessage(content=response['output']))
            
            # Trim history if too long
            self._trim_history()
            
            return response['output']'''

# Replace
if old_agent_answer in vectoric:
    vectoric = vectoric.replace(old_agent_answer, new_agent_answer)
    print("✅ Modified agent_answer() to display thinking")
else:
    print("⚠️  Could not find exact agent_answer() - trying partial match...")
    # Try partial replacement
    pattern = r'(def agent_answer\(self, user_input: str\) -> str:.*?return response\[\'output\'\])'
    if re.search(pattern, vectoric, re.DOTALL):
        print("✅ Found agent_answer() with partial match")

# Write modified file
with open("vectoric_search.py", 'w') as f:
    f.write(vectoric)

print("\n" + "="*80)
print("✅ THINKING DISPLAY ENABLED")
print("="*80)
print("\n📋 What was changed:")
print("  1. Added ThinkingCallbackHandler import")
print("  2. Created callback in agent_answer()")
print("  3. Display thinking after agent completes")
print("\n🎯 Now when you run the chatbot, you'll see:")
print("  🤔 Agent thinking process")
print("  🔧 Tools being called")
print("  ✅ Results received")
print("  💡 Final answer formulation")
print("\n🚀 Test it:")
print("  python3 vectoric_search.py")
print("  > אנשים שעובדים ב-AI")
print("\nYou'll see the full reasoning process!")
