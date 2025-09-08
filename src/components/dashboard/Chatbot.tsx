// src/components/dashboard/Chatbot.tsx

"use client"; // Important: This makes it a Client Component

import { useState } from 'react';
import { MessageSquare, X, Send } from 'lucide-react';

export default function Chatbot() {
  // State to manage if the chat window is open or closed
  const [isOpen, setIsOpen] = useState<boolean>(false);

  return (
    <div className="fixed bottom-6 right-6 z-50">
      {/* Conditionally render the chat window if isOpen is true */}
      {isOpen && (
        <div className="bg-white w-80 h-96 rounded-lg shadow-xl flex flex-col">
          {/* Header */}
          <div className="bg-green-600 text-white p-3 flex justify-between items-center rounded-t-lg">
            <h3 className="font-semibold">AI Farmer Assistant</h3>
            <button onClick={() => setIsOpen(false)} className="hover:opacity-75">
              <X size={20} />
            </button>
          </div>

          {/* Message Area */}
          <div className="flex-1 p-4 overflow-y-auto text-sm text-gray-700">
            <div className="bg-gray-100 p-2 rounded-lg self-start max-w-xs">
              Hello! How can I help you with your farm today?
            </div>
          </div>

          {/* Input Area */}
          <div className="p-2 border-t flex items-center">
            <input
              type="text"
              placeholder="Ask a question..."
              className="flex-1 px-3 py-2 border rounded-full focus:outline-none focus:ring-2 focus:ring-green-500"
            />
            <button className="ml-2 p-2 bg-green-600 text-white rounded-full hover:bg-green-700">
              <Send size={18} />
            </button>
          </div>
        </div>
      )}

      {/* The main button to open/close the chat */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="mt-4 bg-green-600 text-white rounded-full p-4 shadow-lg hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500"
        aria-label="Toggle Chatbot"
      >
        {/* Change icon based on state */}
        {isOpen ? <X size={28} /> : <MessageSquare size={28} />}
      </button>
    </div>
  );
}