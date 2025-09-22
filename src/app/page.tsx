"use client";
import Link from "next/link";
import { SignedIn, SignedOut, UserButton, useUser } from "@clerk/nextjs";
import { useRouter } from "next/navigation";
import { useEffect } from "react";

export default function HomePage() {
  const router = useRouter();
  const { isSignedIn } = useUser();

  useEffect(() => {
    if (isSignedIn) {
      router.replace("/dashboard");
    }
  }, [isSignedIn, router]);

  return (
    <div className="min-h-screen bg-gray-50">
      <nav className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16 items-center">
            <h1 className="text-xl font-semibold">My App</h1>
            
            <div className="flex items-center space-x-4">
              <SignedOut>
                <Link
                  href="/sign-in"
                  className="text-gray-600 hover:text-gray-900 px-3 py-2 rounded-md text-sm font-medium"
                >
                  Sign In
                </Link>
                <Link
                  href="/sign-up"
                  className="bg-blue-600 hover:bg-blue-700 text-white px-3 py-2 rounded-md text-sm font-medium"
                >
                  Sign Up
                </Link>
              </SignedOut>
              
              <SignedIn>
                <Link
                  href="/dashboard"
                  className="text-gray-600 hover:text-gray-900 px-3 py-2 rounded-md text-sm font-medium"
                >
                  Dashboard
                </Link>
                <UserButton afterSignOutUrl="/" />
              </SignedIn>
            </div>
          </div>
        </div>
      </nav>
      
      <main className="max-w-7xl mx-auto py-12 px-4 sm:px-6 lg:px-8">
        <div className="text-center">
          <h1 className="text-4xl font-bold text-gray-900 mb-8">
            Welcome to My App
          </h1>
          
          <SignedOut>
            <p className="text-xl text-gray-600 mb-8">
              Please sign in or create an account to get started.
            </p>
            <div className="space-x-4">
              <Link
                href="/sign-in"
                className="inline-flex items-center px-6 py-3 border border-transparent text-base font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700"
              >
                Sign In
              </Link>
              <Link
                href="/sign-up"
                className="inline-flex items-center px-6 py-3 border border-gray-300 text-base font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50"
              >
                Sign Up
              </Link>
            </div>
          </SignedOut>
        </div>
      </main>
    </div>
  );
}