import { UserButton } from "@clerk/nextjs";
import { currentUser } from "@clerk/nextjs/server";
import { redirect } from "next/navigation";
import Link from "next/link";
import { Leaf, Bot, LayoutDashboard } from "lucide-react";
import type { ReactNode } from "react";
import Image from "next/image";

export default async function DashboardLayout({ children }: { children: ReactNode }) {
  const user = await currentUser();
  if (!user) {
    redirect('/sign-in');
  }

  const fullName = [user.firstName, user.lastName].filter(Boolean).join(' ');

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Top Header */}
      <nav className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16 items-center">
              <Link href="/dashboard" className="flex items-center space-x-2 text-xl font-bold text-green-700">
                  <Image src="/logo.jpg" alt="Logo" width={28} height={28} priority />
                  <span>Disha Kisan</span>
              </Link>
              <div className="flex items-center space-x-4">
              <span className="text-sm text-gray-600 hidden sm:block">
                Welcome, {fullName || user.emailAddresses[0].emailAddress}
              </span>
              <UserButton afterSignOutUrl="/" />
            </div>
          </div>
        </div>
      </nav>

      {/* New Menu Bar */}
      <div className="bg-gray-100 border-b">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
              <div className="flex items-center space-x-4 h-12">
                  <Link href="/dashboard" className="flex items-center px-3 py-2 text-sm font-medium text-gray-700 hover:bg-gray-200 rounded-md">
                      <LayoutDashboard size={18} className="mr-2"/>
                      Dashboard
                  </Link>
                  <Link href="/dashboard/chatbot" className="flex items-center px-3 py-2 text-sm font-medium text-gray-700 hover:bg-gray-200 rounded-md">
                      <Bot size={18} className="mr-2"/>
                      AI Chatbot
                  </Link>
              </div>
          </div>
      </div>

      {/* Page Content */}
      <main className="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
        {children}
      </main>
    </div>
  );
}