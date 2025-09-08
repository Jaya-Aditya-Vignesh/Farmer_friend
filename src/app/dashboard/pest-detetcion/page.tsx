import { ScanLine } from 'lucide-react';
import DashboardCard from '@/components/dashboard/DashboardCard';

export default function PestDetectionPage() {
  return (
    <DashboardCard
      icon={<ScanLine className="w-12 h-12 text-red-500" />}
      title="Pest & Disease Detection"
      description="This is where the pest detection feature will live. You can build a UI here to allow farmers to upload images of their crops."
    >
      <div className="mt-6 p-8 border-2 border-dashed border-gray-300 rounded-lg text-center">
        <p className="text-gray-500">Image Upload Component Goes Here</p>
      </div>
    </DashboardCard>
  );
}