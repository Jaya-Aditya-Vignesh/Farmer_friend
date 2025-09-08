import { redirect } from 'next/navigation';
import { currentUser } from '@clerk/nextjs/server';

// --- Other imports remain the same ---
import FarmMap from '@/components/dashboard/FarmMap';
import WeatherForecast from '@/components/dashboard/WeatherForecast';
// ...etc

// This function fetches from your Python user profile API
async function getUserProfile(userId: string) {
  try {
    // Make sure to use the correct URL for your running Python API
    const res = await fetch(`http://127.0.0.1:5005/api/profile/${userId}`, { cache: 'no-store' });

    // If user doesn't exist in DB yet, API returns 404
    if (res.status === 404) {
      return null;
    }
    if (!res.ok) {
        throw new Error("Failed to fetch user profile");
    }
    return res.json();
  } catch (error) {
    console.error(error);
    return null; // Return null on error to trigger setup
  }
}

export default async function DashboardPage() {
  const user = await currentUser();
  if (!user) {
    // This redirect is handled by the layout, but it's good practice
    redirect('/sign-in');
  }

  const userProfile = await getUserProfile(user.id);

  // --- GATEKEEPER LOGIC ---
  // If the profile is missing or incomplete, redirect to the setup page.
  if (!userProfile || !userProfile.location || !userProfile.chosenCrop) {
    redirect('/dashboard/setup');
  }

  // If the code reaches here, the profile is complete.
  // Destructure for easier use.
  const { location, chosenCrop } = userProfile;

  // --- RENDER THE FULL DASHBOARD (same as before) ---
  return (
    <div className="relative">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* All your dashboard components that use `location` and `chosenCrop` go here */}
      </div>
    </div>
  );
}