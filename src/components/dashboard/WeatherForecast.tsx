interface LocationData {
  lat: number;
  lng: number;
  name: string;
}

interface WeatherForecastProps {
  location: LocationData;
}

interface WeatherDay {
  day: string;
  temp: string;
  condition: string;
}

export default function WeatherForecast({ location }: WeatherForecastProps) {
  const mockWeather: WeatherDay[] = [
    { day: "Today", temp: "31°C", condition: "Sunny" },
    { day: "Tomorrow", temp: "32°C", condition: "Partly Cloudy" },
    { day: "Wed", temp: "29°C", condition: "Showers" },
  ];

  return (
    <div className="mt-4 space-y-3">
      {mockWeather.map((weather) => (
        <div key={weather.day} className="flex justify-between items-center p-2 bg-gray-50 rounded-md">
          <p className="font-medium text-gray-700">{weather.day}</p>
          <p className="text-gray-600">{weather.condition}</p>
          <p className="font-semibold text-gray-800">{weather.temp}</p>
        </div>
      ))}
    </div>
  );
}