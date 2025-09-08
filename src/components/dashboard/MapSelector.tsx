"use client";

import { MapContainer, TileLayer, Marker, useMapEvents } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';
import { useState, useEffect } from 'react';

// Fix for default icon not loading in Next.js
delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.7.1/dist/images/marker-icon-2x.png',
  iconUrl: 'https://unpkg.com/leaflet@1.7.1/dist/images/marker-icon.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.7.1/dist/images/marker-shadow.png',
});

interface MapSelectorProps {
  location: { lat: number; lng: number } | null;
  setLocation: (location: { name: string; lat: number; lng: number }) => void;
}

const LocationMarker = ({ setLocation }: { setLocation: MapSelectorProps['setLocation'] }) => {
  const [position, setPosition] = useState<L.LatLngExpression | null>(null);
  const map = useMapEvents({
    click: async (e) => {
      const { lat, lng } = e.latlng;
      setPosition(e.latlng);
     
      try {
        // UPDATED PORT to 5000
        const res = await fetch(`http://127.0.0.1:5000/api/reverse-geocode?lat=${lat}&lon=${lng}`);
        if (!res.ok) throw new Error("Could not find address.");
        const data = await res.json();
        setLocation({ name: data.name, lat, lng });
      } catch (err) {
        setLocation({ name: `Lat: ${lat.toFixed(4)}, Lng: ${lng.toFixed(4)}`, lat, lng });
      }
    }
  });
  return position === null ? null : <Marker position={position}></Marker>;
};

export default function MapSelector({ location, setLocation }: MapSelectorProps) {
  const defaultCenter: L.LatLngExpression = location ? [location.lat, location.lng] : [13.0827, 80.2707];
  
  return (
    <div className="h-96 w-full">
      <MapContainer
        center={defaultCenter}
        zoom={13}
        scrollWheelZoom={false}
        style={{ height: '100%', width: '100%' }}
      >
        <TileLayer
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        />
        <LocationMarker setLocation={setLocation} />
      </MapContainer>
    </div>
  );
}