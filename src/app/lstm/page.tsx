"use client";

import { useState } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";

type LstmResponse = { finishP50: number };

export default function LstmPage() {
  const [raceId, setRaceId] = useState("942");
  const [driverId, setDriverId] = useState("18");
  const [currentLap, setCurrentLap] = useState("51");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<LstmResponse | null>(null);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setResult(null);

    const r = Number(raceId);
    const d = Number(driverId);
    const lap = Number(currentLap);

    if (!Number.isFinite(r) || !Number.isFinite(d) || !Number.isFinite(lap)) {
      setError("Please enter valid numbers for race, driver, and lap.");
      return;
    }

    setLoading(true);
    try {
      const res = await fetch("https://f1-race-prediction-production.up.railway.app/lstm/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ raceId: r, driverId: d, currentLap: lap }),
      });

      if (!res.ok) {
        const text = await res.text();
        throw new Error(`Backend error (${res.status}): ${text}`);
      }

      const data = (await res.json()) as LstmResponse;
      setResult(data);
    } catch (err: any) {
      setError(err.message ?? "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex w-full justify-center p-4 md:p-8">
      <Card className="w-full max-w-xl">
        <CardHeader>
          <CardTitle>In-Race LSTM Finish Predictor</CardTitle>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-6">
            <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
              <div className="space-y-2">
                <Label htmlFor="raceId">Race ID</Label>
                <Input
                  id="raceId"
                  type="number"
                  value={raceId}
                  onChange={(e) => setRaceId(e.target.value)}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="driverId">Driver ID</Label>
                <Input
                  id="driverId"
                  type="number"
                  value={driverId}
                  onChange={(e) => setDriverId(e.target.value)}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="currentLap">Current Lap</Label>
                <Input
                  id="currentLap"
                  type="number"
                  value={currentLap}
                  onChange={(e) => setCurrentLap(e.target.value)}
                />
              </div>
            </div>

            <Button type="submit" disabled={loading}>
              {loading ? "Predicting..." : "Predict Finish"}
            </Button>

            <Separator />

            {error && <div className="text-sm text-red-500">Error: {error}</div>}

            {result && (
              <div className="space-y-2">
                <div className="text-sm text-muted-foreground">
                  Model output (LSTM prediction for this lap):
                </div>
                <div className="text-2xl font-semibold">
                  Expected finishing position:{" "}
                  <span className="font-mono">P{result.finishP50.toFixed(2)}</span>
                </div>
              </div>
            )}
          </form>
        </CardContent>
      </Card>
    </div>
  );
}
