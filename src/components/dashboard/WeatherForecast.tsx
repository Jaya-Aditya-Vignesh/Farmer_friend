import { useEffect, useState } from "react";

interface LocationData {
  lat: number;
  lng: number;
  name:string;
}

interface WeatherForecastProps {
  location: LocationData;
}

// The component's internal state for a single day's forecast
interface WeatherDay {
  day: string;
  temp: string;
  condition: string;
}

// Type definitions for the relevant parts of the WeatherAPI.com response
interface WeatherApiCondition {
  text: string;
}

interface WeatherApiDayData {
  avgtemp_c: number;
  condition: WeatherApiCondition;
}

interface WeatherApiForecastDay {
  date: string;
  day: WeatherApiDayData;
}

export default function WeatherForecast({ location }: WeatherForecastProps) {
  const [weather, setWeather] = useState<WeatherDay[] | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchWeather() {
      if (!location.lat || !location.lng) return;

      setLoading(true);
      setError(null);

      // Securely get the API key from environment variables
      const apiKey = process.env.NEXT_PUBLIC_WEATHERAPI_KEY;

      if (!apiKey) {
        setError("WeatherAPI.com key is not configured.");
        setLoading(false);
        return;
      }

      try {
        // Use the WeatherAPI.com forecast endpoint
        const res = await fetch(
          `https://api.weatherapi.com/v1/forecast.json?key=${apiKey}&q=${location.lat},${location.lng}&days=5`
        );

        if (!res.ok) {
           const errorData = await res.json();
           throw new Error(errorData.error.message || `Failed to fetch weather data (Status: ${res.status})`);
        }

        const data = await res.json();

        if (data && data.forecast && data.forecast.forecastday) {
          // Process the API data to fit our component's needs
          const dailyData = data.forecast.forecastday.map((forecastDay: WeatherApiForecastDay): WeatherDay => ({
            // Add 'T00:00:00' to ensure Date parsing is consistent across timezones
            day: new Date(forecastDay.date + 'T00:00:00').toLocaleDateString("en-US", { weekday: "long" }),
            temp: `${Math.round(forecastDay.day.avgtemp_c)}°C`,
            condition: forecastDay.day.condition.text,
          }));

          setWeather(dailyData);
        } else {
          throw new Error("No forecast data available in the response");
        }
      } catch (err: any) {
        setError(err.message || "Could not load weather.");
        setWeather(null);
      } finally {
        setLoading(false);
      }
    }

    fetchWeather();
  }, [location.lat, location.lng]);

  if (loading) {
    return <div className="mt-4 text-gray-500">Loading weather...</div>;
  }

  if (error) {
    return <div className="mt-4 text-red-500">Error: {error}</div>;
  }

  if (!weather || weather.length === 0) {
    return <div className="mt-4 text-gray-500">No weather data to display.</div>;
  }

  return (
    <div className="mt-4 space-y-3">
       <h3 className="text-lg font-semibold text-gray-800">5-Day Forecast for {location.name}</h3>
      {weather.map((day) => (
        <div
          key={day.day}
          className="flex justify-between items-center p-2 bg-gray-50 rounded-md"
        >
          <p className="font-medium text-gray-700 w-1/3">{day.day}</p>
          <p className="text-gray-600 text-center w-1/3">{day.condition}</p>
          <p className="font-semibold text-gray-800 text-right w-1/3">
            {day.temp}
          </p>
        </div>
      ))}
    </div>
  );
}