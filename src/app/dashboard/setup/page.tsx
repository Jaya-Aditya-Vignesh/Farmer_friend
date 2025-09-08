"use client";

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { useUser } from '@clerk/nextjs';
import { MapPin, Sparkles, CheckCircle } from 'lucide-react';
import dynamic from 'next/dynamic';

const MapSelector = dynamic(() => import('@/components/dashboard/MapSelector'), { ssr: false });

interface LocationData {
  name: string;
  lat: number;
  lng: number;
}

export default function SetupPage() {
  const router = useRouter();
  const { user } = useUser();

  const [step, setStep] = useState(1);
  const [location, setLocation] = useState<LocationData | null>(null);
  const [language, setLanguage] = useState('en');
  const [recommendedCrop, setRecommendedCrop] = useState<string | null>(null);

  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [showMap, setShowMap] = useState(false);

  useEffect(() => {
    if (user) {
      // UPDATED PORT
      fetch(`http://127.0.0.1:5000/api/profile/${user.id}`)
        .then(res => res.ok ? res.json() : null)
        .then(data => {
          if (data) {
            setLocation(data.location || null);
            setLanguage(data.language || 'en');
            setRecommendedCrop(data.chosenCrop || null);
          }
        });
    }
  }, [user]);

  const handleGetLocation = () => {
    if (!navigator.geolocation) return setError("Geolocation is not supported.");
    setIsLoading(true);
    setError('');
    navigator.geolocation.getCurrentPosition(
      async (position) => {
        const { latitude, longitude } = position.coords;
        try {
          // UPDATED PORT
          const res = await fetch(`http://127.0.0.1:5000/api/reverse-geocode?lat=${latitude}&lon=${longitude}`);
          if (!res.ok) throw new Error("Could not find address.");
          const data = await res.json();
          setLocation({ name: data.name, lat: latitude, lng: longitude });
        } catch (err) {
          setLocation({ name: `Lat: ${latitude.toFixed(4)}, Lng: ${longitude.toFixed(4)}`, lat: latitude, lng: longitude });
        } finally {
          setIsLoading(false);
        }
      },
      () => {
        setError("Unable to retrieve location. Please grant permission.");
        setIsLoading(false);
      }
    );
  };

  const handleSaveLocationAndProceed = async () => {
    if (!user || !location) return setError("Please set a location first.");
    setIsLoading(true);
    setError('');
    try {
      const profileData = { language, location };
      // UPDATED PORT
      await fetch(`http://127.0.0.1:5000/api/profile/${user.id}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(profileData),
      });
      setStep(2);
      handleRecommendCrop(location); // Pass location directly
    } catch (err: any) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleRecommendCrop = async (loc: LocationData) => {
    if (!loc) return;
    setIsLoading(true);
    setError('');
    try {
      // UPDATED PORT
      const res = await fetch(`http://127.0.0.1:5000/api/recommend-crop?lat=${loc.lat}&lon=${loc.lng}`);
      if (!res.ok) throw new Error("Could not get a recommendation.");
      const data = await res.json();
      setRecommendedCrop(data.recommended_crop);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleFinishSetup = async () => {
    if (!user || !recommendedCrop) return;
    setIsLoading(true);
    try {
      const profileData = { chosenCrop: recommendedCrop };
      // UPDATED PORT
      await fetch(`http://127.0.0.1:5000/api/profile/${user.id}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(profileData),
      });
      router.push('/dashboard');
    } catch (err: any) {
      setError(err.message);
      setIsLoading(false);
    }
  };

  return (
    <div className="max-w-xl mx-auto bg-white p-8 rounded-lg shadow text-gray-800">
      <h1 className="text-2xl font-bold text-black">Profile Setup</h1>
      {step === 1 && (
        <div className="space-y-6 mt-6">
          <p className="text-gray-600">First, let's set your primary farm location.</p>
          <div>
            <label className="block text-sm font-medium text-black">Set Your Farm's Location</label>
            <div className="mt-2 flex items-center space-x-2">
              <button type="button" onClick={handleGetLocation} disabled={isLoading} className="flex-shrink-0 flex items-center px-4 py-2 border rounded-md text-sm text-white bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400">
                <MapPin className="mr-2 h-5 w-5" />
                Get Auto Location
              </button>
              <button type="button" onClick={() => setShowMap(!showMap)} className="flex-shrink-0 px-4 py-2 border rounded-md text-sm text-gray-700 bg-gray-200 hover:bg-gray-300">
                {showMap ? 'Hide Map' : 'Select Manually'}
              </button>
            </div>
            {showMap && <div className="mt-4"><MapSelector location={location} setLocation={setLocation} /></div>}
            <div className="mt-4 w-full p-2 border rounded-md bg-gray-50 text-sm font-semibold">{location ? location.name : 'Location will appear here...'}</div>
          </div>
          {location && (
            <button onClick={handleSaveLocationAndProceed} disabled={isLoading} className="w-full flex justify-center py-2 px-4 border rounded-md text-sm text-white bg-green-600 hover:bg-green-700 disabled:bg-gray-400">
              {isLoading ? 'Saving...' : 'Next: Recommend Crop'}
            </button>
          )}
          {error && <p className="text-sm text-red-600 mt-2">{error}</p>}
        </div>
      )}
      {step === 2 && (
         <div className="space-y-6 mt-6">
            <div className="p-4 bg-green-50 border border-green-200 rounded-md flex items-center">
                <CheckCircle className="h-5 w-5 mr-3 text-green-600" />
                <p className="text-sm font-medium text-green-800">Location saved: {location?.name}</p>
            </div>
            <p className="text-gray-600">Now, let's find the best crop for your land.</p>
            <div>
                <label className="block text-sm font-medium text-black">Get a Crop Recommendation</label>
                <div className="mt-2 flex items-center space-x-4">
                    <button type="button" onClick={() => handleRecommendCrop(location!)} disabled={isLoading} className="flex items-center px-4 py-2 border rounded-md text-sm text-white bg-purple-600 hover:bg-purple-700 disabled:bg-gray-400">
                        <Sparkles className="mr-2 h-5 w-5" />
                        {isLoading ? 'Analyzing...' : 'Recommend a Crop'}
                    </button>
                    <div className="w-full p-2 border rounded-md bg-gray-50 text-sm font-bold">{recommendedCrop || '...'}</div>
                </div>
            </div>
            {error && <p className="text-sm text-red-600">{error}</p>}
            <button onClick={handleFinishSetup} disabled={isLoading || !recommendedCrop} className="w-full flex justify-center py-2 px-4 border rounded-md text-sm text-white bg-green-600 hover:bg-green-700 disabled:bg-gray-400">
                {isLoading ? 'Saving...' : 'Finish Setup'}
            </button>
         </div>
      )}
    </div>
  );
}