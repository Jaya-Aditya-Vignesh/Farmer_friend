"use client";

import { useState } from 'react';
import { useRouter } from 'next/navigation';
// You might need to get the user ID from the Clerk session hook
import { useUser } from '@clerk/nextjs';

export default function SetupPage() {
  const router = useRouter();
  const { user } = useUser(); // Clerk hook to get user info on the client
  const [locationName, setLocationName] = useState('');
  const [crop, setCrop] = useState('');
  const [language, setLanguage] = useState('en');
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!user) {
      setError("User not found. Please sign in again.");
      return;
    }
    if (!locationName || !crop) {
      setError("Please fill in all fields.");
      return;
    }
    setIsLoading(true);

    // Note: In a real app, you'd get lat/lng from a geocoding API
    // For now, we'll use placeholder coordinates.
    const profileData = {
      language,
      chosenCrop: crop,
      location: {
        name: locationName,
        lat: 0.0, // Placeholder
        lng: 0.0, // Placeholder
      },
    };

    try {
      const response = await fetch(`/api/profile/${user.id}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(profileData),
      });

      if (!response.ok) {
        throw new Error('Failed to save profile.');
      }

      // On success, redirect to the main dashboard
      router.push('/dashboard');

    } catch (err: any) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="max-w-xl mx-auto bg-white p-8 rounded-lg shadow">
      <h1 className="text-2xl font-bold mb-2 text-gray-800">Welcome!</h1>
      <p className="text-gray-600 mb-6">Let's set up your farm profile to get started.</p>

      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label htmlFor="location" className="block text-sm font-medium text-gray-700">Farm Location (City, State)</label>
          <input
            id="location"
            type="text"
            value={locationName}
            onChange={(e) => setLocationName(e.target.value)}
            placeholder="e.g., Chennai, Tamil Nadu"
            className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm"
          />
        </div>
        <div>
          <label htmlFor="crop" className="block text-sm font-medium text-gray-700">Primary Crop</label>
          <input
            id="crop"
            type="text"
            value={crop}
            onChange={(e) => setCrop(e.target.value)}
            placeholder="e.g., Rice"
            className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm"
          />
        </div>
        <div>
          <label htmlFor="language" className="block text-sm font-medium text-gray-700">Preferred Language</label>
          <select
            id="language"
            value={language}
            onChange={(e) => setLanguage(e.target.value)}
            className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm"
          >
            <option value="en">English</option>
            <option value="ta">Tamil</option>
            <option value="hi">Hindi</option>
          </select>
        </div>

        {error && <p className="text-sm text-red-600">{error}</p>}

        <button
          type="submit"
          disabled={isLoading}
          className="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-green-600 hover:bg-green-700 disabled:bg-gray-400"
        >
          {isLoading ? 'Saving...' : 'Save and Continue'}
        </button>
      </form>
    </div>
  );
}