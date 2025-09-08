"use client";

import { useState, useRef, useEffect } from 'react';
import { Send, Mic } from 'lucide-react';

export default function ChatbotPage() {
  const [messages, setMessages] = useState<{ type: string; text: string }[]>([
    { type: 'bot', text: 'Hello! Ask me a question or press the mic to speak.' }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isListening, setIsListening] = useState(false);
  const recognitionRef = useRef<any>(null);
  const audioRef = useRef<HTMLAudioElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const SpeechRecognition = window.SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (SpeechRecognition) {
      const recognitionInstance = new SpeechRecognition();
      recognitionInstance.continuous = false;
      recognitionInstance.lang = 'en-IN';
      recognitionInstance.onresult = (event: any) => {
        const transcript = event.results[0][0].transcript;
        sendMessage(transcript);
        setIsListening(false);
      };
      recognitionInstance.onerror = () => setIsListening(false);
      recognitionRef.current = recognitionInstance;
    }
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleListen = () => {
    const recognition = recognitionRef.current;
    if (!recognition) return alert("Voice recognition not supported.");
    if (isListening) {
      recognition.stop();
    } else {
      setIsListening(true);
      recognition.start();
    }
  };

  const sendMessage = async (text: string) => {
    if (!text.trim()) return;
    setMessages(prev => [...prev, { type: 'user', text }]);
    setInputValue('');

    try {
      // UPDATED PORT
      const response = await fetch('http://127.0.0.1:5000/api/voice-chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text }),
      });
      if (!response.ok) throw new Error("Backend response not ok");

      const audioBlob = await response.blob();
      const audioUrl = URL.createObjectURL(audioBlob);
      setMessages(prev => [...prev, { type: 'bot', text: 'Playing response...' }]);
      if (audioRef.current) {
        audioRef.current.src = audioUrl;
        audioRef.current.play();
      }
    } catch (error) {
      setMessages(prev => [...prev, { type: 'bot', text: 'Sorry, I had trouble responding.' }]);
    }
  };

  const handleTextSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    sendMessage(inputValue);
  };

  return (
    <div className="flex flex-col h-[80vh] bg-white rounded-lg shadow-md">
      <div className="flex-1 p-6 overflow-y-auto space-y-4">
        {messages.map((msg, index) => (
          <div key={index} className={`flex ${msg.type === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`p-3 rounded-lg max-w-lg ${msg.type === 'user' ? 'bg-blue-500 text-white' : 'bg-gray-200 text-gray-800'}`}>
              {msg.text}
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>
      <form onSubmit={handleTextSubmit} className="p-4 border-t flex items-center space-x-2">
        <input
          type="text"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          placeholder={isListening ? "Listening..." : "Type your question here..."}
          className="flex-1 w-full px-4 py-2 border rounded-full focus:outline-none focus:ring-2 focus:ring-green-500"
        />
        <button type="button" onClick={handleListen} className={`p-3 rounded-full transition-colors ${isListening ? 'bg-red-500 text-white animate-pulse' : 'bg-gray-200 hover:bg-gray-300'}`}>
          <Mic size={24} />
        </button>
        <button type="submit" className="p-3 bg-green-600 text-white rounded-full hover:bg-green-700 transition-colors">
          <Send size={24} />
        </button>
      </form>
      <audio ref={audioRef} hidden />
    </div>
  );
}