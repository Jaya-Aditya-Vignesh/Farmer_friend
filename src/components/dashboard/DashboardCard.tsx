import type { ReactNode } from 'react';

interface DashboardCardProps {
  icon?: ReactNode;
  title: string;
  description?: string;
  children?: ReactNode;
  actions?: ReactNode; // Add actions prop
}

export default function DashboardCard({ icon, title, description, children, actions }: DashboardCardProps) {
  return (
    <div className="bg-white overflow-hidden shadow rounded-lg">
      <div className="px-4 py-5 sm:p-6">
        <div className="flex items-start justify-between">
          <div className="flex items-start">
            {icon && <div className="mr-4 flex-shrink-0">{icon}</div>}
            <div>
              <h3 className="text-lg font-medium leading-6 text-gray-900">{title}</h3>
              {description && <p className="mt-1 max-w-2xl text-sm text-gray-500">{description}</p>}
            </div>
          </div>
          {actions && <div>{actions}</div>}
        </div>
        <div className="mt-4">
          {children}
        </div>
      </div>
    </div>
  );
}