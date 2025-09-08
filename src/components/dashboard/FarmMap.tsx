interface LocationData {
  lat: number;
  lng: number;
  name: string;
}

interface FarmMapProps {
  location: LocationData;
}

export default function FarmMap({ location }: FarmMapProps) {
  return (
    <div className="h-64 mt-4 bg-gray-200 rounded-lg flex items-center justify-center">
      <p className="text-gray-500 font-medium">
        Map for {location.name} would be displayed here.
      </p>
    </div>
  );
}