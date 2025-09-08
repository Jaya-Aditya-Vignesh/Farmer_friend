"use client";

import { useState, useRef, useEffect } from 'react';
import { MessageSquare, X, Send, Mic } from 'lucide-react';

export default function Chatbot() {
  const [isOpen, setIsOpen] = useState<boolean>(false);
  const [messages, setMessages] = useState<{ type: string; text: string }[]>([
    { type: 'bot', text: 'Hello! Ask me a question or press the mic to speak.' }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isListening, setIsListening] = useState(false);

  // Use useRef to hold the recognition instance, preventing re-creation on re-renders
  const recognitionRef = useRef<any>(null);

  const audioRef = useRef<HTMLAudioElement>(null);

  // --- FIX IS HERE ---
  // All browser-specific code is moved into this useEffect hook
  useEffect(() => {
    // This code now runs only on the client
    const SpeechRecognition = window.SpeechRecognition || (window as any).webkitSpeechRecognition;

    if (SpeechRecognition) {
      const recognitionInstance = new SpeechRecognition();
      recognitionInstance.continuous = false;
      recognitionInstance.lang = 'en-IN'; // Set language for Indian English

      recognitionInstance.onresult = (event: any) => {
        const transcript = event.results[0][0].transcript;
        sendMessage(transcript);
        setIsListening(false);
      };

      recognitionInstance.onerror = (event: any) => {
        console.error("Speech recognition error:", event.error);
        setIsListening(false);
      };

      // Store the instance in the ref
      recognitionRef.current = recognitionInstance;
    }
  }, []); // The empty dependency array ensures this runs only once when the component mounts

  const handleListen = () => {
    const recognition = recognitionRef.current;
    if (!recognition) {
        alert("Your browser does not support voice recognition.");
        return;
    }
    if (isListening) {
      recognition.stop();
      setIsListening(false);
    } else {
      setIsListening(true);
      recognition.start();
    }
  };

  const sendMessage = async (text: string) => {
    if (!text.trim()) return;

    const newMessages = [...messages, { type: 'user', text }];
    setMessages(newMessages);
    setInputValue('');

    try {
      const response = await fetch('http://127.0.0.1:5006/api/voice-chat', {
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
      console.error("Failed to get audio response:", error);
      setMessages(prev => [...prev, { type: 'bot', text: 'Sorry, I had trouble responding.' }]);
    }
  };

  const handleTextSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    sendMessage(inputValue);
  };

  return (
    <div className="fixed bottom-6 right-6 z-50">
      {isOpen && (
        <div className="bg-white w-80 h-96 rounded-lg shadow-xl flex flex-col">
          <div className="bg-green-600 text-white p-3 flex justify-between items-center rounded-t-lg">
            <h3 className="font-semibold">AI Farmer Assistant</h3>
            <button onClick={() => setIsOpen(false)} className="hover:opacity-75"><X size={20} /></button>
          </div>
          <div className="flex-1 p-4 overflow-y-auto space-y-3">
            {messages.map((msg, index) => (
              <div key={index} className={`p-2 rounded-lg max-w-[80%] ${msg.type === 'user' ? 'bg-blue-100 self-end ml-auto' : 'bg-gray-100 self-start'}`}>
                {msg.text}
              </div>
            ))}
          </div>
          <form onSubmit={handleTextSubmit} className="p-2 border-t flex items-center">
            <input
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              placeholder={isListening ? "Listening..." : "Ask a question..."}
              className="flex-1 px-3 py-2 border rounded-full"
            />
            <button type="button" onClick={handleListen} className={`ml-2 p-2 rounded-full ${isListening ? 'bg-red-500 text-white animate-pulse' : 'bg-gray-200'}`}>
              <Mic size={18} />
            </button>
            <button type="submit" className="ml-2 p-2 bg-green-600 text-white rounded-full"><Send size={18} /></button>
          </form>
        </div>
      )}
      <audio ref={audioRef} hidden />
      <button onClick={() => setIsOpen(!isOpen)} className="mt-4 float-right bg-green-600 text-white rounded-full p-4 shadow-lg hover:bg-green-700">
        {isOpen ? <X size={28} /> : <MessageSquare size={28} />}
      </button>
    </div>
  );
}