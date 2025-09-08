"use client";

import { useState } from 'react';
import { Upload, AlertTriangle } from 'lucide-react';
import DashboardCard from '@/components/dashboard/DashboardCard';

export default function PestDetectionPage() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<{ prediction: string; confidence: string } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
      setResult(null);
      setError(null);
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result as string);
      };
      reader.readAsDataURL(selectedFile);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) return setError("Please select an image file first.");
    setIsLoading(true);
    setResult(null);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      // UPDATED PORT
      const response = await fetch('http://127.0.0.1:5000/api/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error('Prediction failed. Please try again.');
      const data = await response.json();
      setResult(data);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <DashboardCard
      title="Pest & Disease Detection"
      description="Upload an image of a crop leaf to detect pests and diseases."
    >
      <form onSubmit={handleSubmit} className="mt-6">
        <div className="flex items-center justify-center w-full">
          <label className="flex flex-col items-center justify-center w-full h-64 border-2 border-gray-300 border-dashed rounded-lg cursor-pointer bg-gray-50 hover:bg-gray-100">
            {preview ? (
              <img src={preview} alt="Image preview" className="h-full w-full object-contain rounded-lg" />
            ) : (
              <div className="flex flex-col items-center justify-center pt-5 pb-6">
                <Upload className="w-10 h-10 mb-3 text-gray-400" />
                <p className="mb-2 text-sm text-gray-500"><span className="font-semibold">Click to upload</span> or drag and drop</p>
              </div>
            )}
            <input type="file" className="hidden" accept="image/png, image/jpeg" onChange={handleFileChange} />
          </label>
        </div>
        <div className="mt-4 text-center">
          <button type="submit" disabled={!file || isLoading} className="px-6 py-2 text-white bg-green-600 rounded-md hover:bg-green-700 disabled:bg-gray-400">
            {isLoading ? 'Analyzing...' : 'Analyze Image'}
          </button>
        </div>
      </form>
      {error && (
        <div className="mt-6 p-4 bg-red-100 text-red-700 rounded-lg">
          <AlertTriangle className="h-5 w-5 mr-3" />
          <span>{error}</span>
        </div>
      )}
      {result && (
        <div className="mt-6 p-4 bg-blue-100 rounded-lg">
          <h3 className="text-lg font-semibold text-blue-800">Analysis Result</h3>
          <p><strong>Prediction:</strong> {result.prediction}</p>
          <p><strong>Confidence:</strong> {result.confidence}</p>
        </div>
      )}
    </DashboardCard>
  );
}