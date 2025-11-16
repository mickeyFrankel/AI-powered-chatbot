import React, { useState, useEffect, useRef } from 'react'
import './App.css'

const API_URL = 'http://localhost:8000'

function App() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [stats, setStats] = useState(null)
  const [showMenu, setShowMenu] = useState(false)
  const messagesEndRef = useRef(null)
  const fileInputRef = useRef(null)

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
      
      if (error.message.includes('timed out') || error.message.includes('took too long')) {
        errorMsg = '⏱️ השאלה ארכה יותר מדי. נסה לפצל אותה לשאלות קטנות יותר.\n\nQuery took too long. Try breaking it into smaller questions.'
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

  const clearDatabase = async () => {
    if (!confirm('⚠️ This will DELETE ALL contacts! Are you sure?')) return
    
    try {
      setLoading(true)
      const response = await fetch(`${API_URL}/clear-database`, { method: 'POST' })
      const data = await response.json()
      
      if (!response.ok) {
        throw new Error(data.detail || 'Failed to clear database')
      }
      
      alert(`✅ ${data.message}`)
      setMessages([])
      await fetchStats()  // Refresh to show 0 contacts
      setShowMenu(false)
    } catch (error) {
      console.error('Failed to clear database:', error)
      alert(`❌ Error clearing database: ${error.message}`)
    } finally {
      setLoading(false)
    }
  }

  const uploadCSV = async (event) => {
    const file = event.target.files[0]
    if (!file) return

    const formData = new FormData()
    formData.append('file', file)

    try {
      setLoading(true)
      const response = await fetch(`${API_URL}/upload-csv`, {
        method: 'POST',
        body: formData
      })
      
      const data = await response.json()
      alert(`✅ ${data.message}\n\n📊 Added: ${data.documents_added} contacts\n📁 Total in database: ${data.total_contacts}`)
      fetchStats()
      setShowMenu(false)
    } catch (error) {
      console.error('Failed to upload CSV:', error)
      alert('❌ Error uploading CSV')
    } finally {
      setLoading(false)
      // CRITICAL: Reset file input so same file can be uploaded again
      if (fileInputRef.current) {
        fileInputRef.current.value = ''
      }
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
          <div className="header-actions">
            <button className="clear-btn" onClick={clearHistory} title="Clear chat history">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M3 6h18M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2"/>
              </svg>
            </button>
            <button className="menu-btn" onClick={() => setShowMenu(!showMenu)} title="Database settings">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="1"/>
                <circle cx="12" cy="5" r="1"/>
                <circle cx="12" cy="19" r="1"/>
              </svg>
            </button>
            {showMenu && (
              <div className="dropdown-menu">
                <button onClick={() => fileInputRef.current.click()} className="menu-item">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M17 8l-5-5-5 5M12 3v12"/>
                  </svg>
                  Upload CSV
                </button>
                <button onClick={clearDatabase} className="menu-item danger">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M3 6h18M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2M10 11v6M14 11v6"/>
                  </svg>
                  Clear Database
                </button>
              </div>
            )}
          </div>
          <input 
            ref={fileInputRef}
            type="file" 
            accept=".csv" 
            onChange={uploadCSV} 
            style={{display: 'none'}}
          />
        </div>

        <div className="messages">
          {messages.length === 0 && (
            <div className="welcome">
              <h2>👋 Welcome! ברוכים הבאים!</h2>
              <p>Ask me anything about your contacts database</p>
              <div className="examples">
                  {["הטלפון של ועד הבית", "כמה אנשים במאגר?", "phone number of Noah"].map((example, idx) => (
                    <button
                      key={idx}
                      className="example"
                      onClick={() => {
                        setInput(example)
                        setTimeout(handleSend, 0)
                      }}
                      disabled={loading}
                      type="button"
                    >
                      Try: {example}
                    </button>
                  ))}
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
