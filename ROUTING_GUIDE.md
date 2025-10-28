# Smart Chatbot - Query Routing Reference

## FREE Mode (Simple Search)
These queries use FREE simple search (no OpenAI costs):

- `phone number of אבי`
- `אבי אתרוגים`
- `Moishi Chen`
- `050-408-8646`
- `email of David`
- `contact info for ברק`
- `@gmail` (email search)
- Hebrew names: `משה`, `דוד`
- Commands: `load`, `stats`, `history`, `clear`

## AI Mode (Costs Money)
These queries use OpenAI AI (costs ~$0.01-0.05 per query):

- `who is David Ben Gurion?`
- `what is machine learning?`
- `explain the difference between X and Y`
- `compare A to B`
- `summarize this document`
- `how does X work?`
- `why did X happen?`
- `tell me about quantum computing`
- Complex questions with multiple steps

## How It Decides

1. **Pattern matching** - Looks for keywords
2. **Query length** - Short queries (≤3 words) → Simple
3. **Question mark** - Ends with `?` → AI
4. **Complexity** - Multiple conditions → AI

## Examples

```
Query: "phone number of משה"
→ FREE (direct lookup)

Query: "who founded Israel?"  
→ AI (needs reasoning)

Query: "אבי"
→ FREE (name search)

Query: "what are the main contributions of Einstein?"
→ AI (complex question)

Query: "050-408"
→ FREE (phone search)

Query: "compare machine learning to deep learning"
→ AI (analysis needed)
```

## Run It

```bash
python chatbot.py
# or
python smart_chatbot.py
```

The chatbot will show you which mode it's using:
- `⚡ Using simple mode (free, fast)` 
- `🤖 Using AI mode (advanced reasoning)`

## Cost Savings

**Before:** Every query costs money
**After:** Only complex queries cost money

Example session:
- Find 10 contacts → FREE
- Ask "who is Ben Gurion" → $0.01
- Find 5 more contacts → FREE
- Ask "explain machine learning" → $0.02

**Total:** $0.03 instead of $0.15+
