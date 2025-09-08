interface CropCalendarProps {
  crop: string;
}

interface CalendarEvent {
  stage: string;
  date: string;
}

export default function CropCalendar({ crop }: CropCalendarProps) {
  const mockEvents: CalendarEvent[] = [
    { stage: "Planting", date: "Sep 10 - Sep 20" },
    { stage: "Fertilizing", date: "Oct 5" },
    { stage: "Harvesting", date: "Dec 15 - Dec 30" },
  ];

  return (
    <div className="mt-4">
      <ul className="divide-y divide-gray-200">
        {mockEvents.map((event) => (
          <li key={event.stage} className="py-3">
            <p className="font-medium text-gray-800">{event.stage}</p>
            <p className="text-sm text-gray-500">{event.date}</p>
          </li>
        ))}
      </ul>
    </div>
  );
}