import { UserButton } from "@clerk/nextjs";
import { currentUser } from "@clerk/nextjs/server";
import { redirect } from "next/navigation";

export default async function DashboardPage() {
  const user = await currentUser();
  
  if (!user) {
    redirect('/sign-in');
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <nav className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16 items-center">
            <h1 className="text-xl font-semibold">Dashboard</h1>
            <div className="flex items-center space-x-4">
              <span className="text-sm text-gray-600">
                Welcome, {user.firstName || user.emailAddresses[0].emailAddress}
              </span>
              <UserButton afterSignOutUrl="/" />
            </div>
          </div>
        </div>
      </nav>
      
      <main className="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
        <div className="bg-white overflow-hidden shadow rounded-lg">
          <div className="px-4 py-5 sm:p-6">
            <h2 className="text-lg font-medium text-gray-900 mb-4">
              Welcome to your dashboard!
            </h2>
            <p className="text-gray-600 mb-4">
              You have successfully signed in. This is a protected route.
            </p>
            <div className="bg-gray-50 p-4 rounded-lg">
              <h3 className="text-sm font-medium text-gray-900 mb-2">User Information:</h3>
              <p className="text-sm text-gray-600">Email: {user.emailAddresses[0].emailAddress}</p>
              <p className="text-sm text-gray-600">User ID: {user.id}</p>
              {user.firstName && (
                <p className="text-sm text-gray-600">First Name: {user.firstName}</p>
              )}
              {user.lastName && (
                <p className="text-sm text-gray-600">Last Name: {user.lastName}</p>
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}