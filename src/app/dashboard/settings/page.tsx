"use client";
import { useState, useEffect } from "react";
import { useUser } from "@clerk/nextjs";
import DashboardCard from "@/components/dashboard/DashboardCard";

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

export default function SettingsPage() {
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
		<div className="max-w-xl mx-auto mt-10 space-y-6">
			{/* ...existing settings card... */}
			<DashboardCard
				title="Settings"
				description="Manage your account preferences."
			>
				{/* ...other settings can go here... */}
				<div className="mt-4 text-gray-600 text-sm">
					General account settings.
				</div>
			</DashboardCard>
			{/* New card for changing language */}
			<DashboardCard
				title="Change Language"
				description="Select your preferred language for AI responses and dashboard."
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
						className="border rounded px-2 py-1 text-sm"
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
		</div>
	);
}
