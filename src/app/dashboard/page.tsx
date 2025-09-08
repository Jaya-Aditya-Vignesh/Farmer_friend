import { redirect } from 'next/navigation';
import { currentUser } from '@clerk/nextjs/server';
import { MapPin, Target, CloudSun, CalendarDays, ScanLine } from 'lucide-react';
import FarmMap from '@/components/dashboard/FarmMap';
import WeatherForecast from '@/components/dashboard/WeatherForecast';
import CropCalendar from '@/components/dashboard/CropCalendar';
import DashboardCard from '@/components/dashboard/DashboardCard';
import Link from 'next/link';

async function getUserProfile(userId: string) {
  try {
    // UPDATED PORT
    const res = await fetch(`http://127.0.0.1:5000/api/profile/${userId}`, {
      cache: 'no-store'
    });
    if (res.status === 404) return null;
    if (!res.ok) throw new Error("Failed to fetch user profile");
    return res.json();
  } catch (error) {
    console.error("Profile fetch error:", error);
    return null;
  }
}

export default async function DashboardPage() {
  const user = await currentUser();
  if (!user) {
    redirect('/sign-in');
  }

  const userProfile = await getUserProfile(user.id);

  if (!userProfile || !userProfile.location || !userProfile.chosenCrop) {
    redirect('/dashboard/setup');
  }

  const { location, chosenCrop } = userProfile;

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      <div className="lg:col-span-2 space-y-6">
        <DashboardCard
          icon={<MapPin className="text-blue-500" />}
          title="Your Farm Location"
          description={`Currently showing data for ${location.name}.`}
        >
          <FarmMap location={location} />
        </DashboardCard>
        <DashboardCard icon={<CloudSun className="text-orange-500" />} title="Weather Forecast">
          <WeatherForecast location={location} />
        </DashboardCard>
        <DashboardCard
          icon={<ScanLine className="text-red-500" />}
          title="Pest & Disease Detection"
          description="Upload an image of a crop leaf to detect pests and diseases."
        >
          <Link href="/dashboard/pest-detection" className="mt-2 text-sm font-semibold text-green-600 hover:underline">
            Go to Pest Detection Page &rarr;
          </Link>
        </DashboardCard>
      </div>
      <div className="lg:col-span-1 space-y-6">
          <DashboardCard icon={<Target className="text-green-500" />} title="Current Crop" description={`You have selected ${chosenCrop}.`}>
              <Link href="/dashboard/setup" className="mt-2 text-sm font-semibold text-green-600 hover:underline">
                  Change Settings &rarr;
              </Link>
          </DashboardCard>
          <DashboardCard icon={<CalendarDays className="text-purple-500" />} title={`${chosenCrop} Crop Calendar`}>
              <CropCalendar crop={chosenCrop} />
          </DashboardCard>
      </div>
    </div>
  );
}