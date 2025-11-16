/**
 * ThinkingDisplay Component
 * 
 * Displays agent thinking/reasoning process in real-time.
 * Shows step-by-step what the bot is doing.
 * 
 * Usage:
 * <ThinkingDisplay thoughts={thinkingSteps} />
 */

import React from 'react';
import './ThinkingDisplay.css';

export const ThinkingDisplay = ({ thoughts }) => {
  if (!thoughts || thoughts.length === 0) {
    return null;
  }

  const getEmojiForType = (type) => {
    const emojiMap = {
      thinking: '🤔',
      tool_start: '🔧',
      tool_result: '✅',
      error: '❌',
      finish: '💡'
    };
    return emojiMap[type] || '📍';
  };

  return (
    <div className="thinking-container">
      <div className="thinking-header">
        <span className="thinking-icon">🧠</span>
        <h4>Agent Thinking Process</h4>
      </div>
      
      <div className="thinking-steps">
        {thoughts.map((thought, index) => (
          <div key={index} className={`thinking-step ${thought.type}`}>
            <div className="step-header">
              <span className="step-emoji">{thought.emoji || getEmojiForType(thought.type)}</span>
              <span className="step-number">Step {thought.step}</span>
            </div>
            
            {thought.type === 'thinking' && (
              <div className="step-content">
                <p className="thought-text">{thought.thought}</p>
                {thought.tool && (
                  <div className="tool-info">
                    <span className="tool-label">Tool:</span>
                    <code className="tool-name">{thought.tool}</code>
                  </div>
                )}
              </div>
            )}
            
            {thought.type === 'tool_start' && (
              <div className="step-content">
                <p className="tool-message">{thought.message}</p>
              </div>
            )}
            
            {thought.type === 'tool_result' && (
              <div className="step-content">
                <p className="result-message">{thought.message}</p>
                {thought.output && (
                  <div className="result-output">
                    <pre>{thought.output}</pre>
                  </div>
                )}
              </div>
            )}
            
            {thought.type === 'error' && (
              <div className="step-content error">
                <p className="error-message">{thought.message}</p>
              </div>
            )}
            
            {thought.type === 'finish' && (
              <div className="step-content finish">
                <p className="finish-message">{thought.message}</p>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

/**
 * Hook for streaming thinking from API
 */
export const useThinkingStream = () => {
  const [thoughts, setThoughts] = React.useState([]);
  const [isThinking, setIsThinking] = React.useState(false);

  const streamChat = async (message) => {
    setThoughts([]);
    setIsThinking(true);

    try {
      const response = await fetch('http://localhost:8000/chat/stream', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message }),
      });

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6);
            
            if (data === '[DONE]') {
              setIsThinking(false);
              continue;
            }

            try {
              const parsed = JSON.parse(data);
              
              if (parsed.type === 'thinking') {
                setThoughts(prev => [...prev, parsed.data]);
              } else if (parsed.type === 'answer') {
                return parsed.data.text;
              }
            } catch (e) {
              console.error('Parse error:', e);
            }
          }
        }
      }
    } catch (error) {
      console.error('Stream error:', error);
      setIsThinking(false);
    }
  };

  return { thoughts, isThinking, streamChat };
};

/**
 * CSS for ThinkingDisplay
 * Save as ThinkingDisplay.css
 */
export const thinkingCSS = `
.thinking-container {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  border-radius: 12px;
  padding: 20px;
  margin: 16px 0;
  color: white;
  box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}

.thinking-header {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 16px;
  padding-bottom: 12px;
  border-bottom: 2px solid rgba(255,255,255,0.2);
}

.thinking-icon {
  font-size: 24px;
}

.thinking-header h4 {
  margin: 0;
  font-size: 18px;
  font-weight: 600;
}

.thinking-steps {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.thinking-step {
  background: rgba(255,255,255,0.1);
  border-radius: 8px;
  padding: 12px 16px;
  backdrop-filter: blur(10px);
  transition: all 0.3s ease;
  animation: slideIn 0.3s ease;
}

@keyframes slideIn {
  from {
    opacity: 0;
    transform: translateX(-20px);
  }
  to {
    opacity: 1;
    transform: translateX(0);
  }
}

.thinking-step:hover {
  background: rgba(255,255,255,0.15);
  transform: translateX(4px);
}

.step-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
}

.step-emoji {
  font-size: 20px;
}

.step-number {
  font-size: 12px;
  font-weight: 600;
  opacity: 0.8;
}

.step-content {
  margin-left: 28px;
}

.thought-text {
  margin: 0 0 8px 0;
  font-size: 14px;
  line-height: 1.5;
}

.tool-info {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 13px;
  margin-top: 8px;
}

.tool-label {
  opacity: 0.8;
}

.tool-name {
  background: rgba(0,0,0,0.2);
  padding: 2px 8px;
  border-radius: 4px;
  font-family: 'Monaco', monospace;
  font-size: 12px;
}

.result-output {
  background: rgba(0,0,0,0.2);
  border-radius: 4px;
  padding: 8px;
  margin-top: 8px;
  max-height: 100px;
  overflow-y: auto;
}

.result-output pre {
  margin: 0;
  font-size: 12px;
  white-space: pre-wrap;
  word-break: break-all;
}

.error {
  border-left: 3px solid #ff6b6b;
}

.finish {
  border-left: 3px solid #51cf66;
}
`;

/**
 * Example usage in your chat component:
 */
export const ChatExample = () => {
  const { thoughts, isThinking, streamChat } = useThinkingStream();
  const [message, setMessage] = React.useState('');
  const [answer, setAnswer] = React.useState('');

  const handleSend = async () => {
    const result = await streamChat(message);
    setAnswer(result);
  };

  return (
    <div>
      <input 
        value={message}
        onChange={e => setMessage(e.target.value)}
        placeholder="Ask something..."
      />
      <button onClick={handleSend}>Send</button>
      
      {isThinking && <ThinkingDisplay thoughts={thoughts} />}
      
      {answer && <div className="answer">{answer}</div>}
    </div>
  );
};
