# Database Management UI - User Guide

## ✅ New Features Added

Your chatbot interface now has **built-in database management**!

### 🎯 What's New

**1. Menu Button (⋮)**
- Click the three dots in the header
- Access database management options

**2. Upload CSV**
- Upload new contacts directly from the UI
- Automatically fixes phone number format (no more 9.73E+11!)
- Shows success message with contact count

**3. Clear Database**
- Delete all contacts with one click
- Confirmation dialog prevents accidents
- Useful before loading fresh data

**4. Clear Chat History** (existing button)
- Trash icon clears conversation
- Keeps database intact

---

## 📖 How to Use

### Upload New Contacts

1. Click **⋮ menu button** (top right)
2. Click **"Upload CSV"**
3. Select your `.csv` file
4. Wait for confirmation: "✅ Successfully loaded..."
5. Contacts are now searchable!

### Replace All Contacts

1. Click **⋮ menu button**
2. Click **"Clear Database"** 
3. Confirm deletion (⚠️ warning appears)
4. Click **⋮ menu button** again
5. Click **"Upload CSV"**
6. Select new file
7. Done! Fresh database loaded

### Check Contact Count

Look at the badge in the header:
```
1,917 contacts
```

Updates automatically after upload.

---

## 🔧 Phone Number Format

The system **automatically fixes** Excel's scientific notation:

**Excel exports:**  
`9.73E+11` ❌

**System converts to:**  
`0542227884` ✅

**No manual fixing needed!**

---

## 💡 Tips

**For best results:**
- Export from Excel as **CSV UTF-8**
- Use consistent column names (Phone, Email, Name)
- Test with small CSV first (10-20 rows)

**Common column names that work:**
- Phone, Phone 1, Phone Number, Mobile
- Email, E-mail, Email Address
- Name, Full Name, Contact Name

---

## 🎨 UI Features

✅ **Clean dropdown menu** - Modern design  
✅ **Icon-based actions** - Easy to understand  
✅ **Confirmation dialogs** - Prevent accidents  
✅ **Loading indicators** - Know when processing  
✅ **Success messages** - Confirm actions worked  

---

## 🚀 Quick Workflow

**Daily use:**
```
1. Upload CSV → 2. Query contacts → 3. Get answers
```

**Replace database:**
```
1. Clear Database → 2. Upload new CSV → Done
```

**No command line needed!** Everything in the UI.

---

## Restart to See Changes

```bash
./start.sh
```

Then open: **http://localhost:3000**

Click the **⋮** button and try it out! 🎉
