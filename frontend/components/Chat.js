'use client';
// components/Chat.js
import { useState } from 'react';
import styles from './Chat.module.css';

export default function Chat() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  // generate random conversation ID with numbers in string type
  const [conversationID, setConversationID] = useState(
    Math.random().toString().slice(2)
  );

  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMessage = { sender: 'user', text: input };
    setMessages([...messages, userMessage]);

    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ message: input, conversationID: conversationID }),
    });
    const data = await response.json();

    const botMessage = { sender: 'bot', text: data.data.output.content };
    setMessages([...messages, userMessage, botMessage]);
    setInput('');
  };

  return (
    <div className={styles.chatContainer}>
      <div className={styles.messages}>
        {messages.map((msg, index) => (
          <div key={index} className={msg.sender === 'user' ? styles.userMessage : styles.botMessage}>
            {msg.text}
          </div>
        ))}
      </div>
      <div className={styles.inputContainer}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => (e.key === 'Enter' ? sendMessage() : null)}
          className={styles.input}
          placeholder="Type your message..."
        />
        <button onClick={sendMessage} className={styles.button}>Send</button>
      </div>
    </div>
  );
}
