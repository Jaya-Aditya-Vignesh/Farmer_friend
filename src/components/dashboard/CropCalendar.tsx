import { useEffect, useState } from "react";
interface CropCalendarProps {
  crop: string;
  location: { lat: number; lng: number } | null;
}

/**
 * Parses a multi-line string containing a numbered list of crop tasks (with markdown bolding) and their date ranges.
 * @param {string} calendar - The string to parse, expecting lines like "1. **Task** - Start Date - End Date".
 * @returns {{ work: string; dateRange: string }[]} An array of objects for each parsed event.
 */
function trimIntroductoryText(text: string): string {
  // This regular expression looks for the content starting from the first line
  // that begins with "1.".
  // - The 'm' (multiline) flag allows '^' to match the start of any line.
  // - The 's' (dotAll) flag allows '.' to match newline characters.
  const match = text.match(/^\s*1\..*/ms);

  // If a match is found, return the matched part (which is the "1." line and everything after it).
  // Otherwise, return an empty string.
  return match ? match[0] : '';
}
function parseCalendar(calendar: string): { work: string; dateRange: string }[] {
  type CalendarEvent = { work: string; dateRange: string };
  const trimmedText = trimIntroductoryText(calendar);

  // If there's nothing left after trimming, return an empty array.
  if (!trimmedText) {
    return [];
  }

  const lineRegex = /^\d+\.\s+\*\*(.*?)\*\*\s+-\s+(.*)$/;

  // Correctly use the `trimmedText` variable for parsing.
  return trimmedText
    .split("\n")
    .map((line): CalendarEvent | null => {
      const match = line.match(lineRegex);
      if (match) {
        return {
          work: match[1].trim(),
          dateRange: match[2].trim(),
        };
      }
      return null;
    })
    .filter((event): event is CalendarEvent => event !== null);
}
export default function CropCalendar({ crop, location }: CropCalendarProps) {
  const [calendar, setCalendar] = useState<string>("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchCalendar() {
      if (!crop || !location) {
        setCalendar("");
        setLoading(false);
        return;
      }
      setLoading(true);
      try {
        const response = await fetch(
          "http://127.0.0.1:5000/api/crop-calendar",
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              crop,
              lat: location.lat,
              lon: location.lng,
            }),
          }
        );
        if (!response.ok) {
          throw new Error("Failed to fetch crop calendar");
        }
        const data = await response.json();
        setCalendar(data.calendar || "");
      } catch (error) {
        console.error("Error fetching crop calendar:", error);
        setCalendar("Could not load crop calendar. Please try again later.");
      }
      setLoading(false);
    }
    fetchCalendar();
  }, [crop, location]);

  const events = parseCalendar(calendar);

  return (
    <div className="mt-4">
      {loading ? (
        <p>Loading crop calendar...</p>
      ) : events.length > 0 ? (
        <table className="min-w-full border border-gray-300 rounded text-black">
          <thead>
            <tr className="bg-gray-100">
              <th className="py-2 px-4 border-b text-left">Work</th>
              <th className="py-2 px-4 border-b text-left">Date Range</th>
            </tr>
          </thead>
          <tbody>
            {events.map((ev, idx) => (
              <tr key={idx} className="border-b">
                <td className="py-2 px-4 font-semibold">{ev.work}</td>
                <td className="py-2 px-4">{ev.dateRange}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : calendar ? (
        <div className="whitespace-pre-line text-gray-800">{calendar}</div>
      ) : (
        <p className="text-red-500">No crop calendar available.</p>
      )}
    </div>
  );
}