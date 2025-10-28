import React, { useState, useEffect, useRef } from 'react'
import './App.css'

const API_URL = 'http://localhost:8000'

function App() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [stats, setStats] = useState(null)
  const messagesEndRef = useRef(null)

  useEffect(() => {
    fetchStats()
  }, [])

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  const fetchStats = async () => {
    try {
      const response = await fetch(`${API_URL}/stats`)
      const data = await response.json()
      setStats(data)
    } catch (error) {
      console.error('Failed to fetch stats:', error)
    }
  }

  const detectLanguage = (text) => {
    const hebrewPattern = /[\u0590-\u05FF]/
    return hebrewPattern.test(text) ? 'rtl' : 'ltr'
  }

  const handleSend = async () => {
    if (!input.trim() || loading) return

    const userMessage = { role: 'user', content: input, dir: detectLanguage(input) }
    setMessages(prev => [...prev, userMessage])
    setInput('')
    setLoading(true)

    try {
      const response = await fetch(`${API_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: input })
      })

      const data = await response.json()
      
      if (!response.ok) {
        throw new Error(data.detail || 'Failed to get response')
      }

      const assistantMessage = { 
        role: 'assistant', 
        content: data.response,
        dir: detectLanguage(data.response)
      }
      setMessages(prev => [...prev, assistantMessage])
    } catch (error) {
      console.error('Error:', error)
      let errorMsg = 'שגיאה בחיבור לשרת / Error connecting to server'
      
      if (error.message.includes('timed out')) {
        errorMsg = '⏱️ הבקשה ארכה יותר מדי זמן. נסה שאלה פשוטה יותר / Request timed out. Try a simpler query.'
      }
      
      setMessages(prev => [...prev, { 
        role: 'error', 
        content: errorMsg,
        dir: 'rtl'
      }])
    } finally {
      setLoading(false)
    }
  }

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const clearHistory = async () => {
    try {
      await fetch(`${API_URL}/clear`, { method: 'POST' })
      setMessages([])
    } catch (error) {
      console.error('Failed to clear history:', error)
    }
  }

  return (
    <div className="app">
      <div className="chat-container">
        <div className="header">
          <div className="header-content">
            <h1>AI Chatbot</h1>
            <p className="subtitle">מערכת שאלות ותשובות מבוססת בינה מלאכותית</p>
            {stats && (
              <div className="stats">
                <span className="stat-badge">{stats.document_count.toLocaleString()} contacts</span>
              </div>
            )}
          </div>
          <button className="clear-btn" onClick={clearHistory} title="Clear history">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M3 6h18M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2"/>
            </svg>
          </button>
        </div>

        <div className="messages">
          {messages.length === 0 && (
            <div className="welcome">
              <h2>👋 Welcome! ברוכים הבאים!</h2>
              <p>Ask me anything about your contacts database</p>
              <div className="examples">
                <div className="example">Try: "הטלפון של דוד"</div>
                <div className="example">Try: "כמה אנשים במאגר?"</div>
                <div className="example">Try: "phone number of Noah"</div>
              </div>
            </div>
          )}

          {messages.map((msg, idx) => (
            <div key={idx} className={`message ${msg.role}`} dir={msg.dir}>
              <div className="message-content">
                {msg.content}
              </div>
            </div>
          ))}

          {loading && (
            <div className="message assistant">
              <div className="message-content loading">
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        <div className="input-container">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Type your message... הקלד הודעה..."
            rows="1"
            disabled={loading}
            dir={detectLanguage(input)}
          />
          <button 
            onClick={handleSend} 
            disabled={loading || !input.trim()}
            className="send-btn"
          >
            <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
              <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/>
            </svg>
          </button>
        </div>
      </div>
    </div>
  )
}

export default App
