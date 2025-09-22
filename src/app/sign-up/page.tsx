 "use client";
import { useEffect, useState } from "react";
import { SignUp, useUser } from "@clerk/nextjs";

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

export default function SignUpPage() {
  const { isSignedIn, user } = useUser();
  const [showLangSelect, setShowLangSelect] = useState(false);
  const [selectedLang, setSelectedLang] = useState("en");
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (isSignedIn && user) {
      setShowLangSelect(true);
    }
  }, [isSignedIn, user]);

  const handleLangSave = async () => {
    if (!user?.id) return;
    setSaving(true);
    await fetch("http://127.0.0.1:5000/api/profile/" + user.id, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ language: selectedLang }),
    });
    setSaving(false);
    setShowLangSelect(false);
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-md w-full space-y-8">
        <div>
          <h2 className="mt-6 text-center text-3xl font-extrabold text-gray-900">
            Create your account
          </h2>
        </div>
        {!showLangSelect ? (
          <SignUp
            appearance={{
              elements: {
                rootBox: "mx-auto",
                card: "shadow-lg",
              },
            }}
            signInUrl="/sign-in"
            // Only show Google as sign-up method
            routing="path"
            afterSignUpUrl="/sign-up"

          />
        ) : (
          <div className="bg-white p-6 rounded-lg shadow-lg">
            <h3 className="text-lg font-semibold mb-4 text-center">
              Select your preferred language
            </h3>
            <select
              value={selectedLang}
              onChange={(e) => setSelectedLang(e.target.value)}
              className="w-full p-2 border rounded mb-4"
            >
              {LANGUAGES.map((lang) => (
                <option key={lang.code} value={lang.code}>
                  {lang.label}
                </option>
              ))}
            </select>
            <button
              onClick={handleLangSave}
              disabled={saving}
              className="w-full bg-green-600 text-white py-2 rounded hover:bg-green-700 transition"
            >
              {saving ? "Saving..." : "Save Language"}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}