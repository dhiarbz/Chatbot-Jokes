import React, { useState, useEffect, useRef } from 'react'
import "./App.css"
import { IoCodeSlash, IoSend } from 'react-icons/io5'
import { BiPlanet } from 'react-icons/bi'
import { FaRegFileCode } from 'react-icons/fa'
import { TbMessageChatbot } from 'react-icons/tb'

const App = () => {
  const [isChatStarted, setIsChatStarted] = useState(false);
  const [message, setMessage] = useState("");
  const [isResponseScreen, setisResponseScreen] = useState(true);
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const messageEndRef = useRef(null);

  const scrollToBottom = () => {
    messageEndRef.current?.scrollIntoView({behavior:"smooth"});
  }

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  const handleSendMessage = async () => {
    if (!message.trim()) return; 

    if (!isChatStarted){
      setIsChatStarted(true);
      setMessages([]);
     }

    const userMessage = { text: message, isUser: true};
    setMessages(prevMessages => [...prevMessages, userMessage]);

    const currentMessage = message;
    setMessage("");
    setIsLoading(true);

    try{
      const response = await fetch('http://127.0.0.1:5000/chat',{
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({prompt: currentMessage}),
      });
      if(!response.ok){
        throw new Error('Gagal mendapatkan renspon dari server');
      }
      
      const data = await response.json();
      const botMessage = {text: data.response, isUser:false};
      setMessages(prevMessages => [...prevMessages, botMessage]);
    }catch(error){
      console.error("Error:", error);
      const errorMessage = {text: 'Maaf,Terjadi kesalahan. Coba lagi nanti!', isUser: false};
      setMessages(prevMessages => [...prevMessages, errorMessage]);
    }finally{
      setIsLoading(false);
    }
  };

  const handleNewChat = () => {
      setIsChatStarted(true);
      setMessages([{ text: "Halo! Saya MemeAI, siap memberikan lelucon.", isUser: false }]);
  }

  const handleCardClick = (promptText) => {
      setIsChatStarted(true);
      setMessage();
      handleSendMessage(promptText);
  }

  return (
    <div className="app-container">
      {isChatStarted ? (
        <div className='chat-container'>
          <div className="header">
            <h2>MemeAI</h2>
            <button className='new-chat-btn' onClick={() => setIsChatStarted(false)}>
              <IoCodeSlash className="icon" /> New Chat
            </button>
          </div>
          
          <div className="messages">
            { messages.map((msg, index) => (
                <div 
                  key={index} 
                  className={`message ${msg.isUser ? 'user' : 'bot'}`}
                >
                  {msg.text}
                </div>
              ))}
              {isLoading && (
                <div className="message bot">
                  <i>Bot sedang mengetik...</i>
                </div>
              )}
              <div ref={messageEndRef} />
          </div>
        </div>
      ) : (
        <div className="welcome-screen">
          <h1><TbMessageChatbot className="chatbot-icon" /> MemeAI</h1>
          <div className="card-container">
                <div className="card" onClick={() => handleCardClick("Buat lelucon tentang programmer")}>
                    <p>Buat lelucon tentang programmer</p>
                    <FaRegFileCode className="card-icon" />
                </div>
                <div className="card" onClick={() => handleCardClick("Kenapa ikan tidak bisa tidur?")}>
                    <p>Kenapa ikan tidak bisa tidur?</p>
                    <FaRegFileCode className="card-icon" />
                </div>
                <div className="card" onClick={() => handleCardClick("Apa bedanya apel dan upil?")}>
                    <p>Apa bedanya apel dan upil?</p>
                    <FaRegFileCode className="card-icon" />
                </div>
                <div className="card" onClick={() => handleNewChat()}>
                    <p>Mulai percakapan baru</p>
                    <BiPlanet className="card-icon" />
                </div>
          </div>
        </div>
      )}
      
      <div className="input-container">
        <div className="input-box">
          <input 
            type="text" 
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
            placeholder='Write your message here...' 
          />
          {message && (
            <button onClick={handleSendMessage} disabled={isLoading}>
              <IoSend className="send-icon" />
            </button>
          )}
        </div>
        <p className="footer">MemeAI Developed by Dhia Tawakalna.</p>
      </div>
    </div>
  )
}

export default App