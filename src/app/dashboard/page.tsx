"use client";
import { MapPin, Target, CloudSun, CalendarDays, ScanLine } from 'lucide-react';
import FarmMap from '@/components/dashboard/FarmMap';
import WeatherForecast from '@/components/dashboard/WeatherForecast';
import CropCalendar from '@/components/dashboard/CropCalendar';
import DashboardCard from '@/components/dashboard/DashboardCard';
import Link from 'next/link';
import { useState, useEffect } from "react";
import { useUser } from "@clerk/nextjs";
import { useRouter } from 'next/navigation';

const LANGUAGES = [
  { code: "en", label: "English" },
  { code: "hi", label: "Hindi" },
  { code: "ta", label: "Tamil" },
  { code: "te", label: "Telugu" },
  { code: "kn", label: "Kannada" },
  { code: "ml", label: "Malayalam" },
  { code: "bn", label: "Bengali" },
  { code: "gu", label: "Gujarati" },
  { code: "mr", label: "Marathi" },
  { code: "pa", label: "Punjabi" },
];

export default function DashboardPage() {
  const { user, isLoaded } = useUser();
  const router = useRouter();
  const [profile, setProfile] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!isLoaded) return;
    if (!user) {
      router.replace('/sign-in');
      return;
    }
    fetch(`http://127.0.0.1:5000/api/profile/${user.id}`)
      .then(res => res.ok ? res.json() : null)
      .then(data => {
        if (!data || !data.location || !data.chosenCrop) {
          router.replace('/dashboard/setup');
        } else {
          setProfile(data);
        }
        setLoading(false);
      });
  }, [user, isLoaded, router]);

  if (loading || !profile) {
    return (
      <div className="flex items-center justify-center h-screen">
        <span className="text-gray-500 text-lg">Loading dashboard...</span>
      </div>
    );
  }

  const { location, chosenCrop } = profile;

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
        <DashboardCard icon={<ScanLine className="text-yellow-500" />} title="Soil Health Overview">
          <div className="text-gray-600">Soil health data integration coming soon!</div>
          </DashboardCard>
      </div>
      <div className="lg:col-span-1 space-y-6">
        <DashboardCard icon={<Target className="text-green-500" />} title="Current Crop" description={`You have selected ${chosenCrop}.`}>
          <Link href="/dashboard/setup" className="mt-2 text-sm font-semibold text-green-600 hover:underline">
            Change Settings &rarr;
          </Link>
        </DashboardCard>
        <DashboardCard icon={<CalendarDays className="text-purple-500" />} title={`${chosenCrop} Crop Calendar`}>
          <CropCalendar crop={chosenCrop} location={location} />
        </DashboardCard>
        <LanguageSettingsCard />
      </div>
    </div>
  );
}

function LanguageSettingsCard() {
  const { user } = useUser();
  const [language, setLanguage] = useState("en");
  const [saving, setSaving] = useState(false);
  const [success, setSuccess] = useState(false);

  useEffect(() => {
    if (user?.id) {
      fetch(`http://127.0.0.1:5000/api/profile/${user.id}`)
        .then((res) => (res.ok ? res.json() : null))
        .then((data) => {
          if (data?.language) setLanguage(data.language);
        });
    }
  }, [user]);

  const handleLanguageChange = async (e: React.ChangeEvent<HTMLSelectElement>) => {
    const newLang = e.target.value;
    setLanguage(newLang);
    setSaving(true);
    setSuccess(false);
    await fetch(`http://127.0.0.1:5000/api/profile/${user?.id}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ language: newLang }),
    });
    setSaving(false);
    setSuccess(true);
    setTimeout(() => setSuccess(false), 2000);
  };

  return (
    <DashboardCard
      title="Language Settings"
      description="Change your preferred language for AI responses and dashboard."
    >
      <div className="flex items-center space-x-2">
        <label
          htmlFor="language"
          className="text-sm font-medium text-gray-700 mr-2"
        >
          Language:
        </label>
        <select
          id="language"
          value={language}
          onChange={handleLanguageChange}
          disabled={saving}
          className="border text-black rounded px-2 py-1 text-sm"
        >
          {LANGUAGES.map((lang) => (
            <option key={lang.code} value={lang.code}>
              {lang.label}
            </option>
          ))}
        </select>
        {saving && (
          <span className="text-xs text-gray-400 ml-2">Saving...</span>
        )}
        {success && (
          <span className="text-xs text-green-600 ml-2">Saved!</span>
        )}
      </div>
    </DashboardCard>
  );
}