"use client";

import { useState, useRef, useEffect } from 'react';
import { useUser } from '@clerk/nextjs';
import { MessageSquare, X, Send, Mic, MicOff, Loader2, Volume2, VolumeX, Play, Pause, RotateCcw } from 'lucide-react';

interface Message {
  type: 'user' | 'bot';
  text: string;
  timestamp: Date;
  isVoiceInput?: boolean;
  audioUrl?: string;
  hasAudio?: boolean;
}

export default function Chatbot() {
  const { isLoaded, user } = useUser();
  const [isOpen, setIsOpen] = useState<boolean>(false);
  const [messages, setMessages] = useState<Message[]>([
    {
      type: 'bot',
      text: 'Hello! I\'m your AI Farmer Assistant. Ask me anything about farming or press the mic to speak.',
      timestamp: new Date()
    }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isListening, setIsListening] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [currentlyPlaying, setCurrentlyPlaying] = useState<number | null>(null);
  const [audioEnabled, setAudioEnabled] = useState(true);
  const [playbackSpeed, setPlaybackSpeed] = useState(1);

  const recognitionRef = useRef<any>(null);
  const audioRefs = useRef<Map<number, HTMLAudioElement>>(new Map());
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const SpeechRecognition = window.SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (SpeechRecognition) {
      const recognitionInstance = new SpeechRecognition();
      recognitionInstance.continuous = false;
      recognitionInstance.lang = 'en-IN';
      recognitionInstance.interimResults = false;

      recognitionInstance.onstart = () => setIsListening(true);
      recognitionInstance.onresult = (event: any) => {
        const transcript = event.results[0][0].transcript;
        setInputValue(transcript);
        setIsListening(false);
        // Auto-send voice input
        setTimeout(() => sendMessage(transcript, true), 100);
      };
      recognitionInstance.onerror = () => {
        setIsListening(false);
        setMessages(prev => [...prev, {
          type: 'bot',
          text: 'Sorry, I couldn\'t understand your voice. Please try again.',
          timestamp: new Date()
        }]);
      };
      recognitionInstance.onend = () => setIsListening(false);
      recognitionRef.current = recognitionInstance;
    }
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleListen = () => {
    const recognition = recognitionRef.current;
    if (!recognition) {
      alert("Voice recognition not supported in this browser.");
      return;
    }

    if (isListening) {
      recognition.stop();
    } else {
      recognition.start();
    }
  };

  const sendMessage = async (text: string, isVoiceInput: boolean = false) => {
    if (!text.trim()) return;

    if (!isLoaded || !user?.id) {
      setMessages(prev => [...prev, {
        type: 'bot',
        text: 'Please log in to use the chatbot.',
        timestamp: new Date()
      }]);
      return;
    }

    const userMessage: Message = {
      type: 'user',
      text,
      timestamp: new Date(),
      isVoiceInput
    };
    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      const requestAudio = isVoiceInput || audioEnabled;
      const response = await fetch('http://127.0.0.1:5000/api/voice-chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: text,
          userId: user.id,
          voiceInput: isVoiceInput,
          requestAudio: requestAudio
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error('Backend error:', response.status, errorText);
        throw new Error(`Server error: ${response.status}`);
      }

      const contentType = response.headers.get('content-type');

      if (contentType?.includes('audio')) {
        const audioBlob = await response.blob();
        const audioUrl = URL.createObjectURL(audioBlob);

        const botMessage: Message = {
          type: 'bot',
          text: '🎵 Audio response ready.',
          timestamp: new Date(),
          audioUrl,
          hasAudio: true
        };

        setMessages(prev => [...prev, botMessage]);

        // Auto-play for voice inputs
        if (isVoiceInput && audioEnabled) {
          setTimeout(() => {
            playAudio(messages.length, audioUrl);
          }, 500);
        }

      } else {
        const jsonResponse = await response.json();
        const botMessage: Message = {
          type: 'bot',
          text: jsonResponse.text || 'I received your message but couldn\'t generate a response.',
          timestamp: new Date()
        };
        setMessages(prev => [...prev, botMessage]);
      }
    } catch (error) {
      console.error("Failed to get response:", error);
      setMessages(prev => [...prev, {
        type: 'bot',
        text: 'Sorry, I\'m having trouble connecting. Please try again.',
        timestamp: new Date()
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  const playAudio = (messageIndex: number, audioUrl: string) => {
    // Stop any currently playing audio
    audioRefs.current.forEach((audio, index) => {
      if (index !== messageIndex) {
        audio.pause();
        audio.currentTime = 0;
      }
    });
    setCurrentlyPlaying(null);

    let audio = audioRefs.current.get(messageIndex);
    if (!audio) {
      audio = new Audio(audioUrl);
      audio.playbackRate = playbackSpeed;
      audioRefs.current.set(messageIndex, audio);

      audio.onended = () => setCurrentlyPlaying(null);
      audio.onerror = () => {
        setCurrentlyPlaying(null);
        setMessages(prev => prev.map((msg, idx) =>
          idx === messageIndex ? { ...msg, text: msg.text + ' (Audio failed)' } : msg
        ));
      };
    }

    if (currentlyPlaying === messageIndex) {
      audio.pause();
      setCurrentlyPlaying(null);
    } else {
      audio.play().then(() => {
        setCurrentlyPlaying(messageIndex);
      }).catch(() => setCurrentlyPlaying(null));
    }
  };

  const handleSpeedChange = (newSpeed: number) => {
    setPlaybackSpeed(newSpeed);
    audioRefs.current.forEach(audio => {
      audio.playbackRate = newSpeed;
    });
  };

  const handleTextSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (inputValue.trim() && !isLoading) {
      sendMessage(inputValue, false);
    }
  };

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
      hour12: true
    });
  };

  return (
    <div className="fixed bottom-6 right-6 z-50">
      {/* Chat Window */}
      {isOpen && (
        <div className="bg-white w-96 h-[32rem] rounded-2xl shadow-2xl flex flex-col mb-4 border border-gray-200 overflow-hidden">
          {/* Header */}
          <div className="bg-gradient-to-r from-green-600 to-green-700 text-black p-3 flex justify-between items-center">
            <div>
              <h3 className="font-bold text-lg">AI Farmer Assistant</h3>
              <div className="flex items-center space-x-2 text-green-100 text-xs mt-1">
                <span>Speed:</span>
                <select
                  value={playbackSpeed}
                  onChange={(e) => handleSpeedChange(parseFloat(e.target.value))}
                  className="bg-green-500 text-black text-xs rounded px-1 border-none outline-none"
                >
                  <option value={0.75}>0.75x</option>
                  <option value={1}>1x</option>
                  <option value={1.25}>1.25x</option>
                  <option value={1.5}>1.5x</option>
                </select>
              </div>
            </div>

            <div className="flex items-center space-x-2">
              <button
                onClick={() => setAudioEnabled(!audioEnabled)}
                className={`p-1 rounded-full transition-colors ${
                  audioEnabled ? 'bg-green-500 text-black' : 'bg-white/20 text-green-100'
                }`}
                title={audioEnabled ? 'Disable audio' : 'Enable audio'}
              >
                {audioEnabled ? <Volume2 size={14} /> : <VolumeX size={14} />}
              </button>

              <button
                onClick={() => setIsOpen(false)}
                className="hover:bg-white/10 rounded-full p-1 transition-colors"
              >
                <X size={18} />
              </button>
            </div>
          </div>

          {/* Message Area */}
          <div className="flex-1 p-3 overflow-y-auto space-y-3 bg-gray-50">
            {messages.map((msg, index) => (
              <div key={index} className={`flex ${msg.type === 'user' ? 'justify-end' : 'justify-start'}`}>
                <div className="max-w-[85%]">
                  <div className={`p-3 rounded-2xl shadow-sm ${
                    msg.type === 'user' 
                      ? `bg-blue-600 text-black rounded-tr-md ${msg.isVoiceInput ? 'border-2 border-blue-400' : ''}` 
                      : 'bg-white text-black border border-gray-200 rounded-tl-md'
                  }`}>
                    <div className="flex items-start justify-between">
                      <p className={`text-sm whitespace-pre-wrap flex-1 ${msg.type === 'user' ? 'text-black' : ''}`}>{msg.text}</p>

                      {msg.isVoiceInput && msg.type === 'user' && (
                        <Mic size={12} className="ml-2 text-blue-200 flex-shrink-0" />
                      )}

                      {msg.type === 'bot' && msg.hasAudio && msg.audioUrl && (
                        <div className="flex items-center space-x-1 ml-2">
                          <button
                            onClick={() => playAudio(index, msg.audioUrl!)}
                            className="p-1 rounded-full bg-green-100 hover:bg-green-200 transition-colors"
                            title={currentlyPlaying === index ? 'Pause' : 'Play'}
                          >
                            {currentlyPlaying === index ?
                              <Pause size={10} className="text-green-700" /> :
                              <Play size={10} className="text-green-700" />
                            }
                          </button>
                        </div>
                      )}
                    </div>
                  </div>
                  <div className={`mt-1 flex items-center justify-between text-xs text-gray-500 ${
                    msg.type === 'user' ? 'flex-row-reverse' : ''
                  }`}>
                    <span>{formatTime(msg.timestamp)}</span>
                    {msg.type === 'bot' && currentlyPlaying === index && (
                      <span className="text-green-600 font-medium">Playing...</span>
                    )}
                  </div>
                </div>
              </div>
            ))}

            {isLoading && (
              <div className="flex justify-start">
                <div className="bg-white border border-gray-200 rounded-2xl rounded-tl-md p-3 shadow-sm">
                  <div className="flex items-center space-x-2">
                    <Loader2 className="h-4 w-4 animate-spin text-green-600" />
                    <span className="text-sm text-gray-600">AI is thinking...</span>
                  </div>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Input Area */}
          <div className="p-3 border-t bg-white">
            <form onSubmit={handleTextSubmit} className="flex items-end space-x-2">
              <div className="flex-1">
                <input
                  type="text"
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  placeholder={isListening ? "Listening..." : "Ask a farming question..."}
                  className="w-full px-3 py-2 text-sm border border-gray-300 rounded-full focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-transparent"
                  disabled={isLoading || isListening}
                />
              </div>

              <button
                type="button"
                onClick={handleListen}
                disabled={isLoading}
                className={`p-2 rounded-full transition-all duration-200 ${
                  isListening 
                    ? 'bg-red-500 text-black animate-pulse' 
                    : 'bg-gray-100 hover:bg-gray-200 text-gray-700'
                } disabled:opacity-50`}
                title={isListening ? "Stop listening" : "Voice input (auto-sends with audio)"}
              >
                {isListening ? <MicOff size={16} /> : <Mic size={16} />}
              </button>

              <button
                type="submit"
                disabled={!inputValue.trim() || isLoading}
                className="p-2 bg-green-600 text-black rounded-full hover:bg-green-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isLoading ? <Loader2 size={16} className="animate-spin" /> : <Send size={16} />}
              </button>
            </form>

            <div className="mt-2 text-xs text-gray-400 text-center">
              Voice auto-sends • Audio: {audioEnabled ? 'ON' : 'OFF'}
            </div>
          </div>
        </div>
      )}

      {/* Toggle Button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="bg-gradient-to-r from-green-600 to-green-700 text-black rounded-full p-4 shadow-xl hover:shadow-2xl hover:from-green-700 hover:to-green-800 transition-all duration-200 transform hover:scale-105"
        title={isOpen ? "Close chat" : "Open AI Assistant"}
      >
        {isOpen ? <X size={28} /> : <MessageSquare size={28} />}
      </button>
    </div>
  );
}