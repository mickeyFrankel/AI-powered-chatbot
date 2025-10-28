# 🤖 Chatbot Project - Quick Start

## 🎯 **Which File Should I Run?**

### **⭐ RECOMMENDED: AI-Powered Chatbot**
```bash
python chatbot_ai_first.py
```
**Best for:** Smart query understanding, works with "phone of Noah", Hebrew, etc.  
**Requires:** OpenAI API key in `.env`  
**Cost:** ~$3/month for typical usage

### **💰 FREE Alternative: Tier-Based Search**
```bash
python chatbot_tiered.py
```
**Best for:** No API costs, shows both exact matches and corrections  
**Requires:** Nothing (100% local)  
**Note:** Term extraction not as smart as AI version

### **🛡️ ORIGINAL: Stable Version**
```bash
python chatbot.py
```
**Best for:** Most stable, original working version  
**Note:** Has the bugs reported (no hierarchical search, etc.)

---

## 🧹 **Clean Up Extra Files**

I created many experimental files while fixing issues. To organize:

```bash
chmod +x cleanup.sh
./cleanup.sh
```

This moves experimental files to `_archive/` folder.

---

## 📊 **Comparison**

| Feature | Original | Tier-Based | AI-First ⭐ |
|---------|----------|------------|-------------|
| Finds "Noah" | ✅ | ✅ | ✅ |
| Finds "phone of Noah" | ❌ | ⚠️ | ✅ |
| Hebrew extraction | ❌ | ⚠️ | ✅ |
| Shows corrections | ❌ | ✅ | ✅ |
| API Cost | Free | Free | ~$3/mo |
| Intelligence | Medium | Medium | High |

---

## 🚀 **My Recommendation**

**Use `chatbot_ai_first.py`**

It solves all issues:
- ✅ Extracts "Noah" from "phone of Noah"
- ✅ Works with Hebrew queries
- ✅ Shows both exact matches and corrections
- ✅ Truly intelligent query understanding
- ✅ Consistent behavior

---

## 📝 **Quick Test**

After choosing your version, test with:
```
🔍 Query: phone of Noah
🔍 Query: Noah  
🔍 Query: Moishi
🔍 Query: phone of אבי אתרוגים
```

All should work correctly in `chatbot_ai_first.py`!

---

## ❓ **Questions?**

- **"Too many files!"** → Run `./cleanup.sh` to organize
- **"Which is best?"** → `chatbot_ai_first.py` (needs API key)
- **"Want free?"** → `chatbot_tiered.py`
- **"Want stable?"** → `chatbot.py` (original)

---

**Quick Decision:**
- Have OpenAI key? → `chatbot_ai_first.py` ⭐
- No API key? → `chatbot_tiered.py`
- Unsure? → `chatbot.py` (original)
