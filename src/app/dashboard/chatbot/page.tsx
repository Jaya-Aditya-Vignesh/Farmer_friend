"use client";

import { useState, useRef, useEffect } from 'react';
import { useUser } from '@clerk/nextjs';
import { Send, Mic, MicOff, Loader2, Volume2, VolumeX, Play, Pause, RotateCcw, Camera, Upload, Image as ImageIcon, X } from 'lucide-react';

interface Message {
  type: 'user' | 'bot';
  text: string;
  timestamp: Date;
  isVoiceInput?: boolean;
  audioUrl?: string;
  hasAudio?: boolean;
  image?: string;
  pestResult?: {
    prediction: string;
    confidence: string;
  };
}

export default function ChatbotPage() {
  const { isLoaded, user } = useUser();
  const [messages, setMessages] = useState<Message[]>([
    {
      type: 'bot',
      text: 'Hello! I\'m your AI Farmer Assistant. Ask me anything about farming, crops, or agriculture. You can type, use voice input, or upload plant images for pest detection.',
      timestamp: new Date()
    }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isListening, setIsListening] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [currentlyPlaying, setCurrentlyPlaying] = useState<number | null>(null);
  const [audioEnabled, setAudioEnabled] = useState(true);
  const [playbackSpeed, setPlaybackSpeed] = useState(1);
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [showImageUpload, setShowImageUpload] = useState(false);

  const recognitionRef = useRef<any>(null);
  const audioRefs = useRef<Map<number, HTMLAudioElement>>(new Map());
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const cameraInputRef = useRef<HTMLInputElement>(null);

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

      recognitionInstance.onerror = (event: any) => {
        console.error('Speech recognition error:', event.error);
        setIsListening(false);
        setMessages(prev => [...prev, {
          type: 'bot',
          text: 'Sorry, I couldn\'t understand your voice. Please try again or type your message.',
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

  const handleImageSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      // Validate file type
      if (!file.type.startsWith('image/')) {
        alert('Please select a valid image file.');
        return;
      }

      // Validate file size (max 10MB)
      if (file.size > 10 * 1024 * 1024) {
        alert('Please select an image smaller than 10MB.');
        return;
      }

      setSelectedImage(file);

      // Create preview
      const reader = new FileReader();
      reader.onload = (e) => {
        setImagePreview(e.target?.result as string);
      };
      reader.readAsDataURL(file);
      setShowImageUpload(true);
    }
  };

  const clearImageSelection = () => {
    setSelectedImage(null);
    setImagePreview(null);
    setShowImageUpload(false);
    if (fileInputRef.current) fileInputRef.current.value = '';
    if (cameraInputRef.current) cameraInputRef.current.value = '';
  };

  // Update the handlePestDetection function in your ChatbotPage component:

// Enhanced handlePestDetection function with better error handling and debugging

// Minimal handlePestDetection - replace your existing function with this:
const handlePestDetection = async () => {
    if (!selectedImage) return;

    if (!isLoaded || !user?.id) {
      setMessages(prev => [...prev, {
        type: 'bot',
        text: 'Please log in to use pest detection.',
        timestamp: new Date()
      }]);
      return;
    }

    const userMessage: Message = {
      type: 'user',
      text: 'Analyzing plant image for pest detection...',
      timestamp: new Date(),
      image: imagePreview!
    };
    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);

    try {
      // Create FormData for image upload
      const formData = new FormData();
      formData.append('file', selectedImage);

      const response = await fetch('http://127.0.0.1:5000/api/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error('Pest detection error:', response.status, errorText);
        throw new Error(`Server error: ${response.status}`);
      }

      // --- THE FIX IS HERE ---
      const result = await response.json();

      // The backend is now guaranteed to send a result, so this check is no longer needed.
      // We can directly use the 'result' object.

      const botMessage: Message = {
        type: 'bot',
        text: `🔍 **Pest Detection Results:**\n\n**Plant/Condition:** ${result.prediction}\n**Confidence:** ${result.confidence}\n\n${getPestAdvice(result.raw_prediction)}`,
        timestamp: new Date(),
        pestResult: result
      };

      setMessages(prev => [...prev, botMessage]);

    } catch (error) {
      console.error("Pest detection failed:", error);
      setMessages(prev => [...prev, {
        type: 'bot',
        text: 'Sorry, I couldn\'t analyze the image. Please check your connection and try again with a clear plant image.',
        timestamp: new Date()
      }]);
    } finally {
      setIsLoading(false);
      clearImageSelection();
    }
  };

//
// Enhanced getPestAdvice function
const getPestAdvice = (prediction: string): string => {
  const advice: { [key: string]: string } = {
    'Apple___Apple_scab': '🍎 **Apple Scab Detected**\n\n' +
      '**Immediate Action:** Apply fungicide treatments (Captan, Myclobutanil) every 10-14 days during wet periods.\n' +
      '**Prevention:** Remove fallen leaves, ensure good air circulation around trees, choose resistant varieties.',

    'Apple___Black_rot': '🍎 **Apple Black Rot Detected**\n\n' +
      '**Immediate Action:** Remove infected fruits and branches immediately. Apply copper-based fungicides during dormant season.\n' +
      '**Prevention:** Prune for air circulation, avoid overhead watering, remove mummified fruits.',

    'Apple___healthy': '🍎 **Healthy Apple Plant**\n\n' +
      '**Status:** Your apple plant looks healthy!\n' +
      '**Maintenance:** Continue regular care and monitoring, balanced fertilization, proper pruning.',

    'Corn_(maize)___healthy': '🌽 **Healthy Corn Plant**\n\n' +
      '**Status:** Your corn plant appears healthy!\n' +
      '**Maintenance:** Maintain proper watering and nutrient management, monitor for common pests.',

    'Grape___healthy': '🍇 **Healthy Grape Plant**\n\n' +
      '**Status:** Your grape plant looks healthy!\n' +
      '**Maintenance:** Keep up with regular pruning and pest monitoring.',

    'Potato___healthy': '🥔 **Healthy Potato Plant**\n\n' +
      '**Status:** Your potato plant appears healthy!\n' +
      '**Maintenance:** Continue monitoring for common potato diseases like blight.',

    'Tomato___healthy': '🍅 **Healthy Tomato Plant**\n\n' +
      '**Status:** Your tomato plant looks healthy!\n' +
      '**Maintenance:** Maintain consistent watering and watch for common tomato issues.'
  };

  return advice[prediction] ||
    `**Analysis Complete**\n\n` +
    `**Condition:** ${prediction.replace(/_/g, ' ')}\n\n` +
    `**Recommendation:** Monitor your plant closely and consult local agricultural experts for specific treatment recommendations.`;
};

// Optional: Add a debug button for testing (remove in production)
const testBackendConnection = async () => {
  try {
    const response = await fetch('http://127.0.0.1:5000/api/health');
    const result = await response.json();
    console.log('Backend health check:', result);

    setMessages(prev => [...prev, {
      type: 'bot',
      text: `**Backend Status**: ${result.status}\n` +
            `**Models Available**:\n` +
            `• PyTorch: ${result.models.pytorch_pest ? '✅' : '❌'}\n` +
            `• TensorFlow: ${result.models.tensorflow_pest ? '✅' : '❌'}\n` +
            `• Crop Model: ${result.models.crop_model ? '✅' : '❌'}`,
      timestamp: new Date()
    }]);
  } catch (error) {
    console.error('Backend connection test failed:', error);
    setMessages(prev => [...prev, {
      type: 'bot',
      text: '❌ **Backend Connection Failed**\n\nCannot connect to the server. Please ensure the backend is running on http://127.0.0.1:5000',
      timestamp: new Date()
    }]);
  }
};


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
        // Handle audio response
        const audioBlob = await response.blob();
        const audioUrl = URL.createObjectURL(audioBlob);

        const botMessage: Message = {
          type: 'bot',
          text: isVoiceInput ? '🎵 Audio response ready. Click play to listen.' : '🎵 Audio response generated. Click play to listen.',
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
        // Handle text response
        const jsonResponse = await response.json();
        const botMessage: Message = {
          type: 'bot',
          text: jsonResponse.text || 'I received your message but couldn\'t generate a proper response.',
          timestamp: new Date()
        };
        setMessages(prev => [...prev, botMessage]);
      }
    } catch (error) {
      console.error("Failed to get response:", error);
      setMessages(prev => [...prev, {
        type: 'bot',
        text: 'Sorry, I\'m having trouble connecting to the server. Please check your connection and try again.',
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

    // Get or create audio element
    let audio = audioRefs.current.get(messageIndex);
    if (!audio) {
      audio = new Audio(audioUrl);
      audio.playbackRate = playbackSpeed;
      audioRefs.current.set(messageIndex, audio);

      audio.onended = () => {
        setCurrentlyPlaying(null);
      };

      audio.onerror = () => {
        setCurrentlyPlaying(null);
        setMessages(prev => prev.map((msg, idx) =>
          idx === messageIndex ? { ...msg, text: msg.text + ' (Audio playback failed)' } : msg
        ));
      };
    }

    // Play or pause audio
    if (currentlyPlaying === messageIndex) {
      audio.pause();
      setCurrentlyPlaying(null);
    } else {
      audio.play().then(() => {
        setCurrentlyPlaying(messageIndex);
      }).catch(err => {
        console.error('Audio play failed:', err);
        setCurrentlyPlaying(null);
      });
    }
  };

  const restartAudio = (messageIndex: number) => {
    const audio = audioRefs.current.get(messageIndex);
    if (audio) {
      audio.currentTime = 0;
      audio.play().then(() => {
        setCurrentlyPlaying(messageIndex);
      });
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

  if (!isLoaded) {
    return (
      <div className="flex items-center justify-center h-[80vh]">
        <div className="flex items-center space-x-2">
          <Loader2 className="h-6 w-6 animate-spin text-green-600" />
          <span className="text-gray-600">Loading chatbot...</span>
        </div>
      </div>
    );
  }

  if (!user) {
    return (
      <div className="flex items-center justify-center h-[80vh] bg-gray-50 rounded-lg">
        <div className="text-center">
          <h2 className="text-xl font-semibold text-gray-800 mb-2">Authentication Required</h2>
          <p className="text-gray-600">Please log in to use the AI Farmer Assistant.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-[80vh] bg-white rounded-xl shadow-lg border border-gray-200 overflow-hidden">
      {/* Header */}
      <div className="bg-gradient-to-r from-green-600 to-green-700 px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-xl font-bold text-white">AI Farmer Assistant</h1>
            <p className="text-green-100 text-sm">Your agricultural companion with voice support & pest detection</p>
          </div>

          {/* Audio Controls */}
          <div className="flex items-center space-x-3">
            {/* Speed Control */}
            <div className="flex items-center space-x-1 text-green-100">
              <span className="text-xs">Speed:</span>
              <select
                value={playbackSpeed}
                onChange={(e) => handleSpeedChange(parseFloat(e.target.value))}
                className="bg-green-500 text-white text-xs rounded px-1 py-0.5 border-none outline-none"
              >
                <option value={0.5}>0.5x</option>
                <option value={0.75}>0.75x</option>
                <option value={1}>1x</option>
                <option value={1.25}>1.25x</option>
                <option value={1.5}>1.5x</option>
                <option value={2}>2x</option>
              </select>
            </div>

            {/* Audio Toggle */}
            <button
              onClick={() => setAudioEnabled(!audioEnabled)}
              className={`p-2 rounded-full transition-colors ${
                audioEnabled ? 'bg-green-500 text-white' : 'bg-white/20 text-green-100'
              }`}
              title={audioEnabled ? 'Disable audio responses' : 'Enable audio responses'}
            >
              {audioEnabled ? <Volume2 size={16} /> : <VolumeX size={16} />}
            </button>
          </div>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 p-6 overflow-y-auto space-y-4 bg-gray-50">
        {messages.map((msg, index) => (
          <div key={index} className={`flex ${msg.type === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[80%] ${msg.type === 'user' ? 'order-last' : ''}`}>
              <div className={`p-4 rounded-2xl shadow-sm ${
                msg.type === 'user' 
                  ? `bg-blue-600 text-black rounded-tr-md ${msg.isVoiceInput ? 'border-2 border-blue-400' : ''}` 
                  : 'bg-white text-black border border-gray-200 rounded-tl-md'
              }`}>
                {/* Image Display */}
                {msg.image && (
                  <div className="mb-3">
                    <img
                      src={msg.image}
                      alt="Uploaded plant image"
                      className="max-w-full h-auto rounded-lg border border-gray-300"
                      style={{ maxHeight: '200px' }}
                    />
                  </div>
                )}

                <div className="flex items-start justify-between">
                  <div className={`whitespace-pre-wrap flex-1 ${msg.type === 'user' ? 'text-black' : ''}`}>
                    {msg.text.includes('**') ? (
                      // Render markdown-style bold text
                      msg.text.split('**').map((part, i) =>
                        i % 2 === 1 ? <strong key={i}>{part}</strong> : part
                      )
                    ) : (
                      msg.text
                    )}
                  </div>

                  {/* Voice Input Indicator */}
                  {msg.isVoiceInput && msg.type === 'user' && (
                    <Mic size={14} className="ml-2 text-blue-200" />
                  )}

                  {/* Audio Controls for Bot Messages */}
                  {msg.type === 'bot' && msg.hasAudio && msg.audioUrl && (
                    <div className="flex items-center space-x-1 ml-3">
                      <button
                        onClick={() => playAudio(index, msg.audioUrl!)}
                        className="p-1 rounded-full bg-green-100 hover:bg-green-200 transition-colors"
                        title={currentlyPlaying === index ? 'Pause audio' : 'Play audio'}
                      >
                        {currentlyPlaying === index ?
                          <Pause size={12} className="text-green-700" /> :
                          <Play size={12} className="text-green-700" />
                        }
                      </button>
                      <button
                        onClick={() => restartAudio(index)}
                        className="p-1 rounded-full bg-gray-100 hover:bg-gray-200 transition-colors"
                        title="Restart audio"
                      >
                        <RotateCcw size={12} className="text-gray-700" />
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

        {/* Loading indicator */}
        {isLoading && (
          <div className="flex justify-start">
            <div className="bg-white border border-gray-200 rounded-2xl rounded-tl-md p-4 shadow-sm">
              <div className="flex items-center space-x-2">
                <Loader2 className="h-4 w-4 animate-spin text-green-600" />
                <span className="text-gray-600">AI is thinking...</span>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Image Upload Modal */}
      {showImageUpload && (
        <div className="absolute inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-xl p-6 max-w-md w-full mx-4">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-semibold text-gray-800">Pest Detection</h3>
              <button
                onClick={clearImageSelection}
                className="text-gray-400 hover:text-gray-600"
              >
                <X size={20} />
              </button>
            </div>

            {imagePreview && (
              <div className="mb-4">
                <img
                  src={imagePreview}
                  alt="Preview"
                  className="w-full h-48 object-cover rounded-lg border"
                />
              </div>
            )}

            <div className="flex space-x-3">
              <button
                onClick={handlePestDetection}
                disabled={isLoading}
                className="flex-1 bg-green-600 text-white py-2 px-4 rounded-lg hover:bg-green-700 transition-colors disabled:opacity-50"
              >
                {isLoading ? (
                  <div className="flex items-center justify-center">
                    <Loader2 className="h-4 w-4 animate-spin mr-2" />
                    Analyzing...
                  </div>
                ) : (
                  'Detect Pests'
                )}
              </button>
              <button
                onClick={clearImageSelection}
                className="px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Input Area */}
      <div className="border-t border-gray-200 p-4 bg-white">
        {/* Image Upload Buttons */}
        <div className="flex space-x-2 mb-3">
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleImageSelect}
            className="hidden"
          />
          <input
            ref={cameraInputRef}
            type="file"
            accept="image/*"
            capture="environment"
            onChange={handleImageSelect}
            className="hidden"
          />

          <button
            onClick={() => fileInputRef.current?.click()}
            className="flex items-center space-x-1 px-3 py-1 bg-blue-100 text-blue-700 rounded-full hover:bg-blue-200 transition-colors text-sm"
          >
            <Upload size={14} />
            <span>Upload Image</span>
          </button>

          <button
            onClick={() => cameraInputRef.current?.click()}
            className="flex items-center space-x-1 px-3 py-1 bg-green-100 text-green-700 rounded-full hover:bg-green-200 transition-colors text-sm"
          >
            <Camera size={14} />
            <span>Take Photo</span>
          </button>
        </div>

        <form onSubmit={handleTextSubmit} className="flex items-end space-x-3">
          <div className="flex-1">
            <textarea
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleTextSubmit(e);
                }
              }}
              placeholder={isListening ? "Listening..." : "Type your farming question or upload a plant image..."}
              className="w-full px-4 py-3 text-black border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-transparent resize-none max-h-32 min-h-[48px]"
              rows={1}
              disabled={isLoading || isListening}
            />
          </div>

          <button
            type="button"
            onClick={handleListen}
            disabled={isLoading}
            className={`p-3 rounded-xl transition-all duration-200 ${
              isListening 
                ? 'bg-red-500 text-white shadow-lg animate-pulse' 
                : 'bg-gray-100 hover:bg-gray-200 text-gray-700'
            } disabled:opacity-50 disabled:cursor-not-allowed`}
            title={isListening ? "Stop listening (auto-sends with audio)" : "Start voice input (auto-sends with audio)"}
          >
            {isListening ? <Mic size={20} /> : <MicOff size={20} />}
          </button>

          <button
            type="submit"
            disabled={!inputValue.trim() || isLoading}
            className="p-3 bg-green-600 text-black rounded-xl hover:bg-green-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:bg-green-600"
            title="Send text message"
          >
            {isLoading ? <Loader2 size={20} className="animate-spin" /> : <Send size={20} />}
          </button>
        </form>

        <div className="mt-2 text-xs text-gray-500 text-center">
          <span className="font-medium">Voice:</span> Auto-sends with audio •
          <span className="font-medium"> Text:</span> Press Enter •
          <span className="font-medium"> Images:</span> Upload for pest detection •
          <span className="font-medium"> Audio:</span> {audioEnabled ? 'Enabled' : 'Disabled'}
        </div>
      </div>
    </div>
  );
}