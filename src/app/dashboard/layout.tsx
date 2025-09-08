import { UserButton } from "@clerk/nextjs"; // Correct import for client components
import { currentUser } from "@clerk/nextjs/server"; // Correct import for server functions
import { redirect } from "next/navigation";
import Link from "next/link";
import { Leaf } from "lucide-react";
import type { ReactNode } from "react";

export default async function DashboardLayout({ children }: { children: ReactNode }) {
  const user = await currentUser();
  if (!user) {
    redirect('/sign-in');
  }

  const fullName = [user.firstName, user.lastName].filter(Boolean).join(' ');

  return (
    <div className="min-h-screen bg-gray-50">
      <nav className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16 items-center">
            <Link href="/dashboard" className="flex items-center space-x-2 text-xl font-bold text-green-700">
              <Leaf size={28} />
              <span>Farmer Companion</span>
            </Link>
            <div className="flex items-center space-x-4">
              <span className="text-sm text-gray-600 hidden sm:block">
                Welcome, {fullName || user.emailAddresses[0].emailAddress}
              </span>
              {/* This line will now work correctly */}
              <UserButton afterSignOutUrl="/" />
            </div>
          </div>
        </div>
      </nav>

      <main className="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
        {children}
      </main>
    </div>
  );
}