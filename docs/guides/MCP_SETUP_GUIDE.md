# 🔌 MCP Server Setup Guide

## What is MCP?

**MCP (Model Context Protocol)** allows Claude Desktop to connect to your chatbot and search your documents directly!

---

## 🚀 Quick Setup (3 Steps)

### **Step 1: Run the Setup Script**

```bash
cd /Users/miryamstessman/Downloads/chatbot
source .venv/bin/activate
python setup_mcp_server.py
```

This will:
1. ✅ Install the MCP package
2. ✅ Configure Claude Desktop
3. ✅ Test the server

### **Step 2: Restart Claude Desktop**

1. **Quit Claude Desktop completely** (Cmd+Q or right-click icon → Quit)
2. **Reopen Claude Desktop**
3. Look for the **🔌 plug icon** in the bottom right

### **Step 3: Use Your Chatbot in Claude!**

Once connected, you can ask Claude things like:
- "Search my contacts for AI experts"
- "Find documents about machine learning"
- "What's in my contacts database?"
- "Search for people in the technology industry"

---

## 📋 What the MCP Server Provides

Your chatbot will be available as **"vectordb"** in Claude with these tools:

### **1. `ping`** - Health Check
```
Check if the server is running and see stats
```

### **2. `stats`** - Database Statistics
```
Get document count and collection info
```

### **3. `ask`** - Question Answering
```
Ask questions and get answers with sources
Example: "Who are my machine learning contacts?"
```

### **4. `search`** - Semantic Search
```
Search for specific information
Example: "Find contacts in Tel Aviv"
```

### **5. `ingest_file`** - Add Documents
```
Load new files into the database
Example: Add a new contacts CSV
```

### **6. `list_sources`** - Show Sources
```
See what files are in the database
```

---

## 🔍 Configuration File Location

```
~/Library/Application Support/Claude/claude_desktop_config.json
```

The setup script creates this configuration:

```json
{
  "mcpServers": {
    "vectordb": {
      "command": "/Users/miryamstessman/Downloads/chatbot/.venv/bin/python",
      "args": [
        "/Users/miryamstessman/Downloads/chatbot/vectordb_MCP_server.py"
      ],
      "env": {
        "PYTHONPATH": "/Users/miryamstessman/Downloads/chatbot",
        "CHROMA_DB_DIR": "/Users/miryamstessman/Downloads/chatbot/chroma_db"
      }
    }
  }
}
```

---

## 🧪 Testing the Server

### **Test 1: Direct Test**
```bash
cd /Users/miryamstessman/Downloads/chatbot
source .venv/bin/activate
python vectordb_MCP_server.py
```

This should start the server in stdio mode (you won't see output, that's normal).
Press Ctrl+C to stop.

### **Test 2: In Claude Desktop**

After restarting Claude, try asking:
```
"Use the vectordb tool to search for contacts"
```

Claude should automatically use your MCP server!

---

## 🐛 Troubleshooting

### **Problem: Server doesn't appear in Claude**

**Solution:**
1. Make sure Claude Desktop is completely quit (not just closed window)
2. Check config file exists:
   ```bash
   cat ~/Library/Application\ Support/Claude/claude_desktop_config.json
   ```
3. Restart Claude Desktop again

### **Problem: "Cannot find module 'vectoric_search'"**

**Solution:**
Make sure your virtual environment has all dependencies:
```bash
source .venv/bin/activate
pip install chromadb sentence-transformers pandas numpy scikit-learn
```

### **Problem: "No documents in database"**

**Solution:**
Load your contacts first:
```bash
python vectoric_search.py
# Then type: load contacts.csv
```

### **Problem: Server crashes on startup**

**Check logs:**
1. Open Claude Desktop
2. Go to: **Claude → Settings → Developer**
3. Click "**View Logs**"
4. Look for "vectordb" errors

---

## 📊 Checking if it Works

### **In Claude Desktop:**

1. Look for **🔌 icon** in bottom right corner
2. Click it - you should see "vectordb" listed
3. If green ✅ = Connected!
4. If red ❌ = Check troubleshooting above

### **Test Query in Claude:**

```
"Can you check my vectordb stats?"
```

Claude should respond with document counts and database info!

---

## 🎯 Example Conversations

Once connected, you can have conversations like:

**You:** "Search my contacts for people in machine learning"

**Claude:** *Uses vectordb search tool* → "I found 15 contacts related to machine learning: ..."

**You:** "Who do I know in Tel Aviv?"

**Claude:** *Uses vectordb search tool* → "Here are your Tel Aviv contacts: ..."

**You:** "What's in my database?"

**Claude:** *Uses vectordb stats tool* → "You have 1,917 documents (contacts) in your database."

---

## 🔄 Updating the Server

If you modify `vectordb_MCP_server.py` or `vectoric_search.py`:

1. **Quit Claude Desktop** (Cmd+Q)
2. **Restart Claude Desktop**
3. Changes will be loaded automatically

No need to run setup again!

---

## 🔐 Security Notes

- ✅ MCP runs locally on your machine
- ✅ No data sent to external servers
- ✅ Only Claude Desktop can access it
- ✅ Your contacts stay private

---

## 📚 Advanced: Manual Configuration

If you want to manually edit the config:

```bash
# Open config file
open ~/Library/Application\ Support/Claude/claude_desktop_config.json

# Edit with your favorite editor
code ~/Library/Application\ Support/Claude/claude_desktop_config.json
```

---

## ✅ Checklist

Before asking Claude to use your chatbot:

- [ ] Python 3.12 virtual environment active
- [ ] ChromaDB installed and working
- [ ] Documents loaded in database (contacts.csv)
- [ ] MCP package installed (`pip install mcp`)
- [ ] Configuration file created
- [ ] Claude Desktop restarted
- [ ] 🔌 icon visible in Claude Desktop

---

## 🎉 Success!

Once you see the green checkmark next to "vectordb" in Claude's MCP menu, you're ready!

**Try asking Claude:**
- "Search my vectordb for AI experts"
- "What documents are in my database?"
- "Find contacts related to machine learning"

**Your chatbot is now integrated with Claude! 🚀**

---

## 📞 Need Help?

If something doesn't work:

1. Check the troubleshooting section above
2. View Claude Desktop logs (Settings → Developer → View Logs)
3. Test the server directly: `python vectordb_MCP_server.py`
4. Make sure all dependencies are installed in your virtual environment

---

**Created:** $(date)
**Location:** `/Users/miryamstessman/Downloads/chatbot/`
