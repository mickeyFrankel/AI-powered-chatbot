CREATE SCHEMA IF NOT EXISTS chatbot;
GRANT ALL PRIVILEGES ON SCHEMA chatbot TO chatbot_user;

CREATE TABLE IF NOT EXISTS chatbot.contacts (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255),
    email VARCHAR(255),
    company VARCHAR(255),
    industry VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS chatbot.data_load_log (
    id SERIAL PRIMARY KEY,
    table_name VARCHAR(255),
    file_path TEXT,
    status VARCHAR(50),
    loaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
