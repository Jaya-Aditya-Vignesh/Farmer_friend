import React from 'react';

// This interface is now more flexible. It can accept data from different sources.
interface LocationData {
  lat?: number;
  lng?: number;
  latitude?: number;
  longitude?: number;
  name?: string;
  location_name?: string;
  mapUrl?: string;
}

interface FarmMapProps {
  location?: LocationData | null;
}

/**
 * A React component that displays a location on an embedded map.
 * It is robust and can handle different location data formats.
 * If a mapUrl is not provided, it will construct one from coordinates.
 */
export default function FarmMap({ location }: FarmMapProps) {
  console.log("FarmMap received location prop:", location);

  // Gracefully handle different key names for coordinates and name.
  const lat = location?.latitude ?? location?.lat;
  const lon = location?.longitude ?? location?.lng;
  const displayName = location?.location_name ?? location?.name;

  // Use the provided mapUrl, or construct one if we have coordinates.
  let finalMapUrl = location?.mapUrl;
  if (!finalMapUrl && lat && lon) {
    // This creates the embeddable URL for OpenStreetMap
    finalMapUrl = `https://www.openstreetmap.org/export/embed.html?bbox=${lon-0.01},${lat-0.01},${lon+0.01},${lat+0.01}&layer=mapnik&marker=${lat},${lon}`;
  }

  // If we still don't have a URL, show the fallback message.
  if (!finalMapUrl) {
    return (
      <div className="h-64 md:h-80 w-full mt-4 rounded-lg border-2 border-dashed border-gray-300 bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <p className="text-gray-500 font-semibold">Map Not Available</p>
          <p className="text-gray-400 mt-1 text-sm">Valid location data is not available.</p>
        </div>
      </div>
    );
  }

  // If we have a valid URL, display the map.
  return (
    <div className="relative h-64 md:h-80 w-full mt-4 rounded-lg overflow-hidden border-2 border-gray-300 shadow-md">
      {displayName && (
        <div className="flex items-center px-4 py-3 bg-yellow-50 border-b-2 border-yellow-400 rounded-t-lg">
          <span className="text-2xl mr-2">📍</span>
          <span className="font-bold text-lg text-black">{displayName}</span>
        </div>
      )}
      <iframe
        width="100%"
        height="100%"
        frameBorder="0"
        scrolling="no"
        src={finalMapUrl}
        title={`Map of ${displayName || 'selected location'}`}
        aria-label={`Map showing the location of ${displayName || 'selected location'}`}
      />
      {/* Pinpoint marker overlay */}
      {(lat && lon) && (
        <div
          className="absolute"
          style={{
            left: '50%',
            top: '50%',
            transform: 'translate(-50%, -100%)',
            pointerEvents: 'none',
            zIndex: 10,
          }}
        >
          <span style={{ fontSize: '2rem', color: 'red', textShadow: '0 0 4px white' }}>📍</span>
        </div>
      )}
    </div>
  );
}
